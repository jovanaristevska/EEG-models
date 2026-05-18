import logging
import sys

import torch

logger = logging.getLogger(__name__)

EEGFM_PATH = 'D:/EEG-FM-Bench'


# def load_eegpt(path: str):
#     try:
#         checkpoint = torch.load(path, map_location='cpu', weights_only=False)
#         state_dict = checkpoint.get('state_dict', checkpoint)
        
#         # Check what keys are in the checkpoint
#         keys = list(state_dict.keys())[:10]
#         logger.info(f"EEGPT checkpoint keys sample: {keys}")
        
#         # Build minimal EEGPT encoder without importing full framework
#         # Use the same architecture as EEGTransformer but standalone
#         import math
        
#         class _PatchEmbed(torch.nn.Module):
#             def __init__(self, patch_size=64, patch_stride=32, 
#                          embed_dim=512, embed_num=4):
#                 super().__init__()
#                 self.patch_size = patch_size
#                 self.patch_stride = patch_stride
#                 self.embed_num = embed_num
#                 self.proj = torch.nn.Conv1d(
#                     1, embed_dim * embed_num,
#                     kernel_size=patch_size,
#                     stride=patch_stride
#                 )
#             def forward(self, x):
#                 # x: (B, C, T)
#                 B, C, T = x.shape
#                 # Process each channel independently
#                 patches = []
#                 for c in range(C):
#                     ch = x[:, c:c+1, :]  # (B, 1, T)
#                     p = self.proj(ch)      # (B, embed_dim*embed_num, n_patches)
#                     patches.append(p)
#                 return torch.stack(patches, dim=1)  # (B, C, embed_dim*embed_num, n_patches)
        
#         class _SimpleEEGPT(torch.nn.Module):
#             def __init__(self, n_outputs=2, n_chans=19, 
#                          embed_dim=512, embed_num=4,
#                          patch_size=64, patch_stride=32,
#                          n_times=1024):
#                 super().__init__()
#                 self.patch_embed = _PatchEmbed(
#                     patch_size=patch_size,
#                     patch_stride=patch_stride,
#                     embed_dim=embed_dim,
#                     embed_num=embed_num
#                 )
#                 n_patches = (n_times - patch_size) // patch_stride + 1
#                 feat_dim = n_chans * embed_dim * embed_num * n_patches
#                 self.classifier = torch.nn.Linear(feat_dim, n_outputs)
                
#             def forward(self, x_or_batch):
#                 if isinstance(x_or_batch, dict):
#                     x = x_or_batch['data']
#                 else:
#                     x = x_or_batch
#                 B = x.shape[0]
#                 features = self.patch_embed(x)  # (B, C, D, P)
#                 flat = features.reshape(B, -1)
#                 return self.classifier(flat)
        
#         model = _SimpleEEGPT(
#             n_outputs=2,
#             n_chans=19,
#             embed_dim=512,
#             embed_num=4,
#             patch_size=64,
#             patch_stride=32,
#             n_times=1024
#         )
        
#         # Try to load any matching weights
#         model_state = model.state_dict()
#         matched = {}
#         for k, v in state_dict.items():
#             # Strip common prefixes
#             key = k
#             for prefix in ('module.', 'model.', 'encoder.', 'target_encoder.'):
#                 if key.startswith(prefix):
#                     key = key[len(prefix):]
#                     break
#             if key in model_state and model_state[key].shape == v.shape:
#                 matched[key] = v
        
#         if matched:
#             model.load_state_dict(matched, strict=False)
#             logger.info(f"EEGPT loaded {len(matched)}/{len(model_state)} weights")
#         else:
#             logger.warning("EEGPT: no weights matched — using random init (Mock-like)")
        
#         model.eval()
#         logger.info("EEGPT model loaded successfully")
#         return model
        
#     except Exception as e:
#         logger.error(f"Failed to load EEGPT: {e}")
#         return None

def load_eegpt(path: str):
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['model_state_dict']

        # Strip 'module.' prefix
        clean = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                clean[k[7:]] = v
            else:
                clean[k] = v

        # Detect dimensions from checkpoint
        total_dim = clean['encoder.summary_token'].shape[2]      # 512
        embed_num = clean['encoder.summary_token'].shape[1]      # 4
        patch_size = clean['encoder.patch_embed.proj.weight'].shape[3]  # 64
        max_chans = clean['encoder.chan_embed.weight'].shape[0]   # 62
        n_blocks = len(set(k.split('.')[2] for k in clean if k.startswith('encoder.blocks.')))  # 8

        logger.info(f"EEGPT arch: total_dim={total_dim}, embed_num={embed_num}, patch_size={patch_size}, max_chans={max_chans}, n_blocks={n_blocks}")

        class _Attention(torch.nn.Module):
            def __init__(self, dim, num_heads=8):
                super().__init__()
                self.num_heads = num_heads
                self.head_dim = dim // num_heads
                self.scale = self.head_dim ** -0.5
                self.qkv = torch.nn.Linear(dim, dim * 3)
                self.proj = torch.nn.Linear(dim, dim)

            def forward(self, x):
                B, N, C = x.shape
                qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)
                attn = (q @ k.transpose(-2, -1)) * self.scale
                attn = attn.softmax(dim=-1)
                x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                return self.proj(x)

        class _MLP(torch.nn.Module):
            def __init__(self, dim, mlp_ratio=4.0):
                super().__init__()
                hidden = int(dim * mlp_ratio)
                self.fc1 = torch.nn.Linear(dim, hidden)
                self.fc2 = torch.nn.Linear(hidden, dim)
                self.act = torch.nn.GELU()

            def forward(self, x):
                return self.fc2(self.act(self.fc1(x)))

        class _Block(torch.nn.Module):
            def __init__(self, dim, num_heads=8):
                super().__init__()
                self.norm1 = torch.nn.LayerNorm(dim)
                self.attn = _Attention(dim, num_heads)
                self.norm2 = torch.nn.LayerNorm(dim)
                self.mlp = _MLP(dim)

            def forward(self, x):
                x = x + self.attn(self.norm1(x))
                x = x + self.mlp(self.norm2(x))
                return x

        class _EEGPTEncoder(torch.nn.Module):
            def __init__(self, total_dim, embed_num, patch_size, n_blocks, num_heads, max_chans):
                super().__init__()
                self.embed_num = embed_num
                self.total_dim = total_dim
                # summary_token: (1, embed_num, total_dim)
                self.summary_token = torch.nn.Parameter(torch.zeros(1, embed_num, total_dim))
                # patch_embed: Conv2d (total_dim, 1, 1, patch_size)
                self.patch_embed = torch.nn.Module()
                self.patch_embed.proj = torch.nn.Conv2d(1, total_dim, kernel_size=(1, patch_size), stride=(1, patch_size//2))
                # chan_embed: Embedding (max_chans, total_dim)
                self.chan_embed = torch.nn.Embedding(max_chans, total_dim)
                self.blocks = torch.nn.ModuleList([_Block(total_dim, num_heads) for _ in range(n_blocks)])
                self.norm = torch.nn.LayerNorm(total_dim)

            def forward(self, x):
                # x: (B, C, T)
                B, C, T = x.shape
                
                # patch_embed expects (B, 1, C, T)
                x_4d = x.unsqueeze(1)  # (B, 1, C, T)
                p = self.patch_embed.proj(x_4d)  # (B, total_dim, C, n_patches)
                B, D, C_out, n_patches = p.shape
                
                # Reshape to (B, C*n_patches, total_dim)
                p = p.permute(0, 2, 3, 1)  # (B, C, n_patches, total_dim)
                
                # Add channel embeddings
                chan_ids = torch.arange(C, device=x.device)
                ce = self.chan_embed(chan_ids)  # (C, total_dim)
                p = p + ce.unsqueeze(0).unsqueeze(2)  # broadcast over patches
                
                p = p.reshape(B, C * n_patches, D)  # (B, C*n_patches, total_dim)
                
                # Prepend summary tokens: (1, embed_num, total_dim) → (B, embed_num, total_dim)
                summary = self.summary_token.expand(B, -1, -1)
                x_seq = torch.cat([summary, p], dim=1)  # (B, embed_num + C*n_patches, total_dim)
                
                for block in self.blocks:
                    x_seq = block(x_seq)
                x_seq = self.norm(x_seq)
                
                # Use mean of summary tokens as representation
                return x_seq[:, :self.embed_num].mean(dim=1)  # (B, total_dim)

        class _EEGPTModel(torch.nn.Module):
            def __init__(self, total_dim, embed_num, patch_size, n_blocks, num_heads, max_chans, n_outputs=2):
                super().__init__()
                self.encoder = _EEGPTEncoder(total_dim, embed_num, patch_size, n_blocks, num_heads, max_chans)
                self.classifier = torch.nn.Sequential(
                    torch.nn.Linear(total_dim, 128),
                    torch.nn.GELU(),
                    torch.nn.Dropout(0.3),
                    torch.nn.Linear(128, n_outputs)
                )

            def forward(self, x_or_batch):
                if isinstance(x_or_batch, dict):
                    x = x_or_batch['data']
                else:
                    x = x_or_batch
                features = self.encoder(x)
                return self.classifier(features)

        model = _EEGPTModel(
            total_dim=total_dim,
            embed_num=embed_num,
            patch_size=patch_size,
            n_blocks=n_blocks,
            num_heads=8,
            max_chans=max_chans,
            n_outputs=2
        )

        # Load encoder weights
        encoder_state = {k[8:]: v for k, v in clean.items() if k.startswith('encoder.')}
        missing, unexpected = model.encoder.load_state_dict(encoder_state, strict=False)
        if missing:
            logger.warning(f"EEGPT encoder missing: {missing}")
        if unexpected:
            logger.warning(f"EEGPT encoder unexpected: {unexpected}")

        # Load classifier weights
        classifier_state = {}
        for k, v in clean.items():
            if k.startswith('classifier.heads.adhd.mlp.'):
                new_key = k.replace('classifier.heads.adhd.mlp.', '')
                classifier_state[new_key] = v

        if classifier_state:
            missing_c, unexpected_c = model.classifier.load_state_dict(classifier_state, strict=False)
            logger.info(f"EEGPT classifier loaded {len(classifier_state)} weights, missing={missing_c}")
        else:
            logger.warning("EEGPT classifier weights not found!")

        model.eval()
        logger.info("EEGPT model loaded successfully")
        return model

    except Exception as e:
        logger.error(f"Failed to load EEGPT: {e}")
        import traceback
        traceback.print_exc()
        return None
    
# def load_eegnet(path: str):
#     try:
#         model = _EEGNetDirect(n_outputs=2, n_chans=19, n_times=1024)
#         checkpoint = torch.load(path, map_location='cpu', weights_only=False)
#         state_dict = checkpoint.get('state_dict', checkpoint)

#         # Strip DDP / wrapper prefixes: module.encoder.* → *
#         clean = {}
#         for k, v in state_dict.items():
#             key = k
#             for prefix in ('module.encoder.', 'module.model.encoder.', 'module.model.',
#                            'module.', 'model.encoder.', 'model.', 'encoder.'):
#                 if key.startswith(prefix):
#                     key = key[len(prefix):]
#                     break
#             clean[key] = v

#         missing, unexpected = model.load_state_dict(clean, strict=False)
#         if missing:
#             logger.warning(f"EEGNet missing {len(missing)} keys — checkpoint may use different arch")
#         model.eval()
#         logger.info("EEGNet model loaded successfully")
#         return model
#     except Exception as e:
#         logger.error(f"Failed to load EEGNet: {e}")
#         return None

# def load_eegnet(path: str):
#     try:
#         checkpoint = torch.load(path, map_location='cpu', weights_only=False)
#         state_dict = checkpoint.get('model_state_dict', checkpoint)
        
#         model = _EEGNetDirect(n_outputs=2, n_chans=19, n_times=1024)
        
#         # Strip 'module.encoder.' prefix
#         clean = {}
#         for k, v in state_dict.items():
#             key = k
#             if key.startswith('module.encoder.'):
#                 key = key[len('module.encoder.'):]
#             clean[key] = v
        
#         # Handle parametrizations for conv_spatial
#         # 'conv_spatial.parametrizations.weight.original' → 'conv_spatial.weight'
#         final = {}
#         for k, v in clean.items():
#             if 'parametrizations.weight.original' in k:
#                 new_key = k.replace('.parametrizations.weight.original', '.weight')
#                 final[new_key] = v
#             else:
#                 final[k] = v
        
#         missing, unexpected = model.load_state_dict(final, strict=False)
#         if missing:
#             logger.warning(f"EEGNet missing {len(missing)} keys: {missing}")
#         if unexpected:
#             logger.warning(f"EEGNet unexpected {len(unexpected)} keys: {unexpected}")
        
#         model.eval()
#         logger.info("EEGNet model loaded successfully")
#         return model
#     except Exception as e:
#         logger.error(f"Failed to load EEGNet: {e}")
#         return None

def load_eegnet(path: str):
    try:
        import braindecode.models
        
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        model = braindecode.models.EEGNet(
            n_outputs=2,
            n_chans=19,
            n_times=1024,
            sfreq=256.0,
        )
        
        # Strip exactly 'module.encoder.' prefix
        clean = {}
        for k, v in state_dict.items():
            if k.startswith('module.encoder.'):
                key = k[len('module.encoder.'):]
                clean[key] = v
        
        missing, unexpected = model.load_state_dict(clean, strict=False)
        if missing:
            logger.warning(f"EEGNet missing {len(missing)} keys")
        
        model.eval()
        logger.info("EEGNet model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load EEGNet: {e}")
        return None


# ── helpers ──────────────────────────────────────────────────────────────────

WINDOW_SAMPLES = 1024


class _SimpleHead(torch.nn.Module):
    """Minimal 2-class linear probe used when we can't reconstruct the full head."""
    def __init__(self, in_features: int, n_classes: int = 2):
        super().__init__()
        self.fc = torch.nn.Linear(in_features, n_classes)

    def forward(self, x):
        # x: (B, T, embed_num, embed_dim) or (B, flat)
        if x.dim() > 2:
            x = x.reshape(x.size(0), -1)
        return self.fc(x)


# class _EEGNetDirect(torch.nn.Module):
#     """
#     EEGNet reimplemented without braindecode, using braindecode EEGNet/EEGNetv4
#     layer names so checkpoints saved from that class load correctly.
#     """
#     def __init__(self, n_outputs=2, n_chans=19, n_times=1024,
#                  F1=8, D=2, kernel_length=64,
#                  pool1_kernel_size=4, pool2_kernel_size=8,
#                  depthwise_kernel_length=16, drop_prob=0.25):
#         super().__init__()
#         F2 = F1 * D
#         # Names match braindecode EEGNet nn.Sequential module names
#         self.conv_temporal = torch.nn.Conv2d(
#             1, F1, (1, kernel_length), bias=False, padding=(0, kernel_length // 2))
#         self.bnorm_temporal = torch.nn.BatchNorm2d(F1, momentum=0.01, eps=1e-3)
#         self.conv_spatial = torch.nn.Conv2d(
#             F1, F1 * D, (n_chans, 1), bias=False, groups=F1)
#         self.bnorm_1 = torch.nn.BatchNorm2d(F1 * D, momentum=0.01, eps=1e-3)
#         self.elu_1 = torch.nn.ELU()
#         self.pool_1 = torch.nn.AvgPool2d((1, pool1_kernel_size))
#         self.drop_1 = torch.nn.Dropout(drop_prob)
#         self.conv_separable_depth = torch.nn.Conv2d(
#             F1 * D, F1 * D, (1, depthwise_kernel_length),
#             bias=False, groups=F1 * D, padding=(0, depthwise_kernel_length // 2))
#         self.conv_separable_point = torch.nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
#         self.bnorm_2 = torch.nn.BatchNorm2d(F2, momentum=0.01, eps=1e-3)
#         self.elu_2 = torch.nn.ELU()
#         self.pool_2 = torch.nn.AvgPool2d((1, pool2_kernel_size))
#         self.drop_2 = torch.nn.Dropout(drop_prob)

#         # final_layer is a sub-Sequential in braindecode — must match key prefix
#         n_out_time = n_times // (pool1_kernel_size * pool2_kernel_size)
#         self.final_layer = torch.nn.Sequential()
#         self.final_layer.add_module(
#             'conv_classifier',
#             torch.nn.Conv2d(F2, n_outputs, (1, n_out_time), bias=True))

#     def forward(self, x):
#         if x.dim() == 3:
#             x = x.unsqueeze(-1)          # (B, C, T, 1) — Ensure4d
#         x = x.permute(0, 3, 1, 2)        # (B, 1, C, T) — dimshuffle
#         x = self.conv_temporal(x)
#         x = self.bnorm_temporal(x)
#         x = self.conv_spatial(x)
#         x = self.bnorm_1(x)
#         x = self.elu_1(x)
#         x = self.pool_1(x)
#         x = self.drop_1(x)
#         x = self.conv_separable_depth(x)
#         x = self.conv_separable_point(x)
#         x = self.bnorm_2(x)
#         x = self.elu_2(x)
#         x = self.pool_2(x)
#         x = self.drop_2(x)
#         x = self.final_layer.conv_classifier(x)  # (B, n_outputs, 1, 1)
#         return x.squeeze(-1).squeeze(-1)          # (B, n_outputs)
class _EEGNetDirect(torch.nn.Module):
    def __init__(self, n_outputs=2, n_chans=19, n_times=1024,
                 F1=8, D=2, kernel_length=64,
                 pool1_kernel_size=4, pool2_kernel_size=8,
                 depthwise_kernel_length=16, drop_prob=0.25):
        super().__init__()
        F2 = F1 * D
        self.n_chans = n_chans
        
        self.conv_temporal = torch.nn.Conv2d(
            1, F1, (1, kernel_length), bias=False, padding=(0, kernel_length // 2))
        self.bnorm_temporal = torch.nn.BatchNorm2d(F1, momentum=0.01, eps=1e-3)
        self.conv_spatial = torch.nn.Conv2d(
            F1, F1 * D, (n_chans, 1), bias=False, groups=F1)
        self.bnorm_1 = torch.nn.BatchNorm2d(F1 * D, momentum=0.01, eps=1e-3)
        self.elu_1 = torch.nn.ELU()
        self.pool_1 = torch.nn.AvgPool2d((1, pool1_kernel_size))
        self.drop_1 = torch.nn.Dropout(drop_prob)
        self.conv_separable_depth = torch.nn.Conv2d(
            F1 * D, F1 * D, (1, depthwise_kernel_length),
            bias=False, groups=F1 * D, padding=(0, depthwise_kernel_length // 2))
        self.conv_separable_point = torch.nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bnorm_2 = torch.nn.BatchNorm2d(F2, momentum=0.01, eps=1e-3)
        self.elu_2 = torch.nn.ELU()
        self.pool_2 = torch.nn.AvgPool2d((1, pool2_kernel_size))
        self.drop_2 = torch.nn.Dropout(drop_prob)

        n_out_time = n_times // (pool1_kernel_size * pool2_kernel_size)
        self.final_layer = torch.nn.Sequential()
        self.final_layer.add_module(
            'conv_classifier',
            torch.nn.Conv2d(F2, n_outputs, (1, n_out_time), bias=True))

    def forward(self, x):
        # x: (B, n_chans, n_times)
        if x.dim() == 3:
            # (B, C, T) → (B, 1, C, T)
            x = x.unsqueeze(1)
        
        # conv_temporal: (B, 1, C, T) → (B, F1, C, T)
        x = self.conv_temporal(x)
        x = self.bnorm_temporal(x)
        
        # conv_spatial: (B, F1, C, T) → (B, F1*D, 1, T)
        x = self.conv_spatial(x)
        x = self.bnorm_1(x)
        x = self.elu_1(x)
        x = self.pool_1(x)
        x = self.drop_1(x)
        
        # separable conv
        x = self.conv_separable_depth(x)
        x = self.conv_separable_point(x)
        x = self.bnorm_2(x)
        x = self.elu_2(x)
        x = self.pool_2(x)
        x = self.drop_2(x)
        
        # classifier
        x = self.final_layer.conv_classifier(x)  # (B, n_outputs, 1, 1)
        return x.squeeze(-1).squeeze(-1)          # (B, n_outputs)


class _EEGPTInferenceModel(torch.nn.Module):
    """Thin wrapper: encoder → flatten → classifier."""
    def __init__(self, encoder, classifier, embed_num: int):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.embed_num = embed_num

    def forward(self, x_or_batch, chan_ids=None):
        # Accept both a raw tensor and the dict {'data': tensor} from inference.py
        if isinstance(x_or_batch, dict):
            x = x_or_batch['data']
        else:
            x = x_or_batch  # (B, 19, 1024)

        if chan_ids is None:
            # EEGTransformer.forward calls chan_ids.to(x) — must be a tensor
            chan_ids = torch.arange(x.size(1), dtype=torch.long)
        features = self.encoder(x, chan_ids=chan_ids)  # (B, T, embed_num*embed_dim)
        logits = self.classifier(features)
        return logits
