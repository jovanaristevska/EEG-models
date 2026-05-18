"""
NeuroGPT Trainer — inherits from ClassicalTrainer.
"""
import logging
import os
from typing import List

import torch
from torch import nn
from datasets import Dataset as HFDataset

from baseline.abstract.adapter import AbstractDataLoaderFactory
from baseline.abstract.classical import ClassicalTrainer
from baseline.neurogpt.neurogpt_config import NeuroGPTConfig
from baseline.neurogpt.neurogpt_adapter import NeuroGPTDataLoaderFactory
from baseline.neurogpt.model import NeuroGPTModel

logger = logging.getLogger('baseline')


class NeuroGPTTrainer(ClassicalTrainer):
    """NeuroGPT trainer that inherits from ClassicalTrainer."""

    def __init__(self, cfg: NeuroGPTConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.dataloader_factory = NeuroGPTDataLoaderFactory(
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers,
            seed=self.cfg.seed,
        )

    def setup_model(self):
        logger.info("Setting up NeuroGPT model architecture...")

        (ds_name, info) = next(iter(self.ds_info.items()))
        n_chans = info['n_ch']
        n_times = info['wnd_sec'] * self.sfreq
        num_classes = info['n_class']
        model_cfg = self.cfg.model

        model = NeuroGPTModel(
            n_chans=n_chans,
            n_times=n_times,
            num_classes=num_classes,
            ds_name=ds_name,
            # Encoder
            n_filters_time=model_cfg.n_filters_time,
            filter_time_length=model_cfg.filter_time_length,
            pool_time_length=model_cfg.pool_time_length,
            pool_time_stride=model_cfg.pool_time_stride,
            drop_prob=model_cfg.drop_prob,
            num_encoder_layers=model_cfg.num_encoder_layers,
            att_heads=model_cfg.att_heads,
            att_drop_prob=model_cfg.att_drop_prob,
            # GPT
            embedding_dim=model_cfg.embedding_dim,
            num_hidden_layers=model_cfg.num_hidden_layers,
            num_attention_heads=model_cfg.num_attention_heads,
            n_positions=model_cfg.n_positions,
            dropout=model_cfg.dropout,
            # Input
            num_chunks=model_cfg.num_chunks,
            chunk_len=model_cfg.chunk_len,
            ft_only_encoder=model_cfg.ft_only_encoder,
        )

        # Load pretrained weights if specified
        if model_cfg.pretrained_path and os.path.exists(model_cfg.pretrained_path):
            model.from_pretrained(model_cfg.pretrained_path)
            logger.info(f"Loaded pretrained weights from {model_cfg.pretrained_path}")
        else:
            logger.info("No pretrained path — training from scratch")

        model = self.apply_lora(model)
        model = model.to(self.device)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[self.local_rank], find_unused_parameters=True
        )

        self.model = model