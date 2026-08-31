"""
NeuroGPT Model — combines EEGConformer encoder + GPT decoder.
Adapted from https://github.com/wenhui0206/NeuroGPT
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from transformers import GPT2Config, GPT2Model

logger = logging.getLogger('baseline')


# ── EEGConformer Encoder ──────────────────────────────────────────────────────

class _PatchEmbedding(nn.Module):
    def __init__(self, n_filters_time, filter_time_length,
                 pool_time_length, stride_avg_pool, drop_prob, num_std_channels):
        super().__init__()
        # Kept as a 1-element Sequential named `shallownet` so its state-dict key
        # (`shallownet.0.*`) matches the pretrained NeuroGPT checkpoint exactly.
        # This is the temporal-only conv — its kernel spans time only, not channels,
        # so it's channel-count-independent and genuinely transfers from pretraining
        # for every montage.
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, n_filters_time, (1, filter_time_length), (1, 1)),
        )

        # Channel-count-agnostic replacement for the old (n_channels, 1) spatial conv.
        # That conv's kernel spanned the full channel dimension, so its weight shape
        # depended on n_channels — meaning it could never load pretrained weights for
        # any dataset in this benchmark, since none match the checkpoint's native 22
        # channels (19/14/8 all mismatch), and was silently reinitialized every run.
        # Instead: a learned per-channel identity embedding (keyed by a canonical
        # channel index, same approach as EEGPT's chan_embed) injects "which electrode
        # is this" information, and `spatial_conv` is a 1x1 conv applied identically to
        # every channel — its weight shape no longer depends on how many channels the
        # input has, so this component is architecturally consistent (if not literally
        # pretrained-loadable, since the checkpoint never had this shape either) across
        # every montage.
        self.chan_embed = nn.Embedding(num_std_channels, n_filters_time)
        self.spatial_conv = nn.Conv2d(n_filters_time, n_filters_time, (1, 1), (1, 1))

        self.post_spatial = nn.Sequential(
            nn.BatchNorm2d(num_features=n_filters_time),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool_time_length), stride=(1, stride_avg_pool)),
            nn.Dropout(p=drop_prob),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(n_filters_time, n_filters_time, (1, 1), stride=(1, 1)),
            Rearrange("b d_model 1 seq -> b seq d_model"),
        )

    def forward(self, x, chan_ids=None):
        x = self.shallownet(x)  # (B, F, n_chans, T') — temporal-only

        n_chans = x.shape[2]
        if chan_ids is None:
            chan_ids = torch.arange(n_chans, device=x.device)
        chan_ids = chan_ids.to(x.device).long()

        embed = self.chan_embed(chan_ids)              # (n_chans, F)
        embed = embed.t().unsqueeze(0).unsqueeze(-1)    # (1, F, n_chans, 1)
        x = x + embed

        x = self.spatial_conv(x)            # (B, F, n_chans, T'), channel-shared weights
        x = x.mean(dim=2, keepdim=True)     # (B, F, 1, T') — collapse the channel dimension

        x = self.post_spatial(x)
        x = self.projection(x)
        return x


class _MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x):
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum("bhqd, bhkd -> bhqk", queries, keys)
        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum("bhal, bhlv -> bhav", att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.projection(out)


class _ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return x + self.fn(x, **kwargs)


class _FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class _TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size, att_heads, att_drop, forward_expansion=4):
        super().__init__(
            _ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                _MultiHeadAttention(emb_size, att_heads, att_drop),
                nn.Dropout(att_drop),
            )),
            _ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                _FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=att_drop),
                nn.Dropout(att_drop),
            )),
        )


class _TransformerEncoder(nn.Sequential):
    def __init__(self, att_depth, emb_size, att_heads, att_drop):
        super().__init__(*[
            _TransformerEncoderBlock(emb_size, att_heads, att_drop)
            for _ in range(att_depth)
        ])


class EEGConformerEncoder(nn.Module):
    """
    EEGConformer encoder from NeuroGPT.
    Input: (batch*chunks, channels, time)
    Output: (batch*chunks, seq, emb_size)
    """
    def __init__(self, n_chans, n_times, n_filters_time=40,
                 filter_time_length=25, pool_time_length=75,
                 pool_time_stride=15, drop_prob=0.5,
                 att_depth=6, att_heads=10, att_drop_prob=0.5,
                 num_std_channels=63):
        super().__init__()
        self.patch_embedding = _PatchEmbedding(
            n_filters_time=n_filters_time,
            filter_time_length=filter_time_length,
            pool_time_length=pool_time_length,
            stride_avg_pool=pool_time_stride,
            drop_prob=drop_prob,
            num_std_channels=num_std_channels,
        )
        self.transformer = _TransformerEncoder(
            att_depth=att_depth,
            emb_size=n_filters_time,
            att_heads=att_heads,
            att_drop=att_drop_prob,
        )

    def forward(self, x, chan_ids=None):
        # x: (batch*chunks, channels, time)
        x = x.unsqueeze(1)                     # (B*C, 1, chans, time)
        x = self.patch_embedding(x, chan_ids)  # (B*C, seq, emb)
        x = self.transformer(x)                # (B*C, seq, emb)
        return x


# ── GPT Decoder ───────────────────────────────────────────────────────────────

class GPTDecoder(nn.Module):
    """GPT decoder from NeuroGPT."""
    def __init__(self, num_hidden_layers=6, num_attention_heads=12,
                 embed_dim=768, n_positions=512, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        config = GPT2Config(
            vocab_size=1,
            n_positions=n_positions,
            n_embd=embed_dim,
            n_layer=num_hidden_layers,
            n_head=num_attention_heads,
            n_inner=embed_dim * 4,
            resid_pdrop=dropout,
            attn_pdrop=dropout,
            embd_pdrop=dropout,
            activation_function='gelu',
        )
        self.transformer = GPT2Model(config=config)

    def forward(self, inputs_embeds, attention_mask=None):
        outputs = self.transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return outputs['last_hidden_state']


# ── Full NeuroGPT Model ───────────────────────────────────────────────────────

class NeuroGPTModel(nn.Module):
    """
    Full NeuroGPT model: EEGConformer encoder + GPT decoder + classifier head.

    Input batch dict keys:
        'data': (B, C, T) EEG tensor
        'montage': list of montage strings
    """
    def __init__(
        self,
        n_chans: int,
        n_times: int,
        num_classes: int,
        ds_name: str,
        # Encoder params
        n_filters_time: int = 40,
        filter_time_length: int = 25,
        pool_time_length: int = 75,
        pool_time_stride: int = 15,
        drop_prob: float = 0.5,
        num_encoder_layers: int = 6,
        att_heads: int = 10,
        att_drop_prob: float = 0.5,
        # GPT params
        embedding_dim: int = 1024,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 8,
        n_positions: int = 512,
        dropout: float = 0.1,
        # Input params
        num_chunks: int = 2,
        chunk_len: int = 512,
        ft_only_encoder: bool = True,
        num_std_channels: int = 63,
    ):
        super().__init__()
        self.ds_name = ds_name
        self.num_chunks = num_chunks
        self.chunk_len = chunk_len
        self.ft_only_encoder = ft_only_encoder

        # Encoder: EEGConformer
        self.encoder = EEGConformerEncoder(
            n_chans=n_chans,
            n_times=chunk_len,
            n_filters_time=n_filters_time,
            filter_time_length=filter_time_length,
            pool_time_length=pool_time_length,
            pool_time_stride=pool_time_stride,
            drop_prob=drop_prob,
            att_depth=num_encoder_layers,
            att_heads=att_heads,
            att_drop_prob=att_drop_prob,
            num_std_channels=num_std_channels,
        )

        # Calculate encoder output dim
        # seq_len after patch embedding
        dummy = torch.zeros(1, 1, n_chans, chunk_len)
        with torch.no_grad():
            dummy_out = self.encoder.patch_embedding(dummy)
        enc_seq_len = dummy_out.shape[1]
        enc_feat_dim = dummy_out.shape[2]  # = n_filters_time
        self.enc_out_dim = enc_seq_len * enc_feat_dim

        # Projection from encoder output to GPT embedding dim
        self.input_proj = nn.Linear(self.enc_out_dim, embedding_dim)

        # GPT decoder
        self.decoder = GPTDecoder(
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            embed_dim=embedding_dim,
            n_positions=n_positions,
            dropout=dropout,
        )

        # Classification head (pooler + classifier)
        self.pooler = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )

    def from_pretrained(self, pretrained_path: str):
        import warnings
        pretrained = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        current = self.state_dict()

        # Key remap: the checkpoint's BatchNorm was originally at shallownet.2.*
        # (part of the old fused shallownet Sequential). After splitting the channel
        # mixing out into its own channel-agnostic module, that BatchNorm now lives at
        # post_spatial.0.* — but its own shape never depended on channel count either
        # way (num_features=n_filters_time, fixed at 40), so this is a safe,
        # shape-preserving rename, not a real architecture change, and should keep
        # loading exactly as it did before the channel-mixing fix.
        key_remap = {
            f'encoder.patch_embedding.shallownet.2.{suffix}':
                f'encoder.patch_embedding.post_spatial.0.{suffix}'
            for suffix in ('weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked')
        }

        # Filter only matching keys with matching shapes
        filtered = {}
        for k, v in pretrained.items():
            k_mapped = key_remap.get(k, k)
            if k_mapped in current:
                if current[k_mapped].shape == v.shape:
                    filtered[k_mapped] = v
                else:
                    warnings.warn(f'Shape mismatch for {k} -> {k_mapped}: {v.shape} vs {current[k_mapped].shape} — skipping')
            else:
                warnings.warn(f'Skipping {k} — not in current model')

        self.load_state_dict(filtered, strict=False)
        logger.info(f"Loaded {len(filtered)}/{len(pretrained)} pretrained weights from {pretrained_path}")

    def forward(self, batch):
        x = batch['data']           # (B, n_chans, T)
        montage = batch.get('montage', ['adhd/10_20'])[0]
        # `chans_id` is produced per-sample by the adapter (canonical channel index per
        # electrode) and is uniform across a batch, since batches are grouped by
        # dataset/montage — take one copy and let the encoder broadcast it.
        chans_id = batch.get('chans_id', None)
        if chans_id is not None:
            chans_id = chans_id[0]
        B, C, T = x.shape

        # Split into chunks: (B, num_chunks, n_chans, chunk_len)
        #chunks = x.unfold(-1, T // self.num_chunks, T // self.num_chunks)
        chunks = x.unfold(-1, self.chunk_len, self.chunk_len)
        # chunks: (B, n_chans, num_chunks, chunk_len) → (B, num_chunks, n_chans, chunk_len)
        chunks = chunks.permute(0, 2, 1, 3).contiguous()

        # Encode each chunk: (B*num_chunks, n_chans, chunk_len)
        B_chunks = B * self.num_chunks
        #x_chunks = chunks.view(B_chunks, C, T // self.num_chunks)
        x_chunks = chunks.view(B_chunks, C, self.chunk_len)
        enc_out = self.encoder(x_chunks, chan_ids=chans_id)        # (B*num_chunks, seq, emb)
        enc_flat = enc_out.reshape(B_chunks, -1)   # (B*num_chunks, seq*emb)

        if self.ft_only_encoder:
            # Fine-tune only encoder — skip GPT
            enc_seq = enc_flat.view(B, self.num_chunks, -1)  # (B, num_chunks, feat)
            pooled = enc_seq.mean(dim=1)                      # (B, feat)
            # Project to classifier input dim
            pooled = nn.functional.adaptive_avg_pool1d(
                pooled.unsqueeze(1), self.classifier[0].in_features
            ).squeeze(1)
            logits = self.classifier(pooled)
            return logits

        # Project to GPT embedding dim
        seq_embeds = self.input_proj(enc_flat)              # (B*num_chunks, embedding_dim)
        seq_embeds = seq_embeds.view(B, self.num_chunks, -1)  # (B, num_chunks, embedding_dim)

        # GPT forward
        attention_mask = torch.ones(B, self.num_chunks, device=x.device, dtype=torch.long)
        gpt_out = self.decoder(seq_embeds, attention_mask)   # (B, num_chunks, embedding_dim)

        # Pool: take last token
        seq_lengths = attention_mask.sum(dim=1) - 1         # (B,)
        pooled = self.pooler(
            gpt_out[torch.arange(B, device=x.device), seq_lengths]
        )                                                    # (B, embedding_dim)

        logits = self.classifier(pooled)                     # (B, num_classes)
        return logits