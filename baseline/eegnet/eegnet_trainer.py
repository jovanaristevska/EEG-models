import logging
from typing import List

import torch
from torch import nn
import braindecode.models
from datasets import Dataset as HFDataset

from baseline.abstract.adapter import AbstractDataLoaderFactory, AbstractDatasetAdapter, StandardEEGChannelsMixin
from baseline.abstract.classical import ClassicalTrainer
from baseline.eegnet.eegnet_config import EegNetConfig


logger = logging.getLogger('baseline')


class EegNetDatasetAdapter(AbstractDatasetAdapter, StandardEEGChannelsMixin):
    """EEGNet dataset adapter — routes EEGNet through the same shared channel
    selection and normalization as EEGPT/NeuroGPT, instead of the raw pass-through
    it used before (which meant EEGNet got no channel mapping and no normalization)."""

    def _setup_adapter(self):
        self.model_name = 'eegnet'
        super()._setup_adapter()

    def get_supported_channels(self) -> List[str]:
        return self.get_standard_eeg_channels()


class EegNetDataLoaderFactory(AbstractDataLoaderFactory):
    def create_adapter(
        self,
        dataset: HFDataset,
        dataset_names: List[str],
        dataset_configs: List[str]
    ) -> EegNetDatasetAdapter:
        return EegNetDatasetAdapter(dataset, dataset_names, dataset_configs)


class EegNetModel(nn.Module):
    def __init__(self, encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = encoder

    def forward(self, batch):
        x = batch['data']
        logits = self.encoder(x)

        return logits

class EegNetTrainer(ClassicalTrainer):
    def __init__(self, cfg: EegNetConfig):
        super().__init__(cfg)

        self.dataloader_factory = EegNetDataLoaderFactory(
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers,
            seed=self.cfg.seed
        )

    def setup_model(self):
        logger.info(f"Setting up eegnet model architecture...")

        (ds_name, info) = next(iter(self.ds_info.items()))

        self.encoder = braindecode.models.EEGNet(
            n_outputs=info['n_class'],
            n_chans=info['n_ch'],
            n_times=info['wnd_sec'] * self.sfreq,
            sfreq=self.sfreq,
            
        )

        model = EegNetModel(self.encoder)
        model = model.to(self.device)

        model = self.maybe_wrap_ddp(model, find_unused_parameters=True)
        logger.info(f"Model setup complete for {list(self.ds_info.keys())}")

        self.model = model

        return model
