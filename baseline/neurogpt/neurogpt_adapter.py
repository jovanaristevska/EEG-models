"""
NeuroGPT Adapter that inherits from AbstractDataLoaderFactory.
"""
import logging
from typing import List
import torch
import numpy as np
from datasets import Dataset as HFDataset
from baseline.abstract.adapter import AbstractDatasetAdapter, AbstractDataLoaderFactory, StandardEEGChannelsMixin

logger = logging.getLogger("baseline")


class NeuroGPTDatasetAdapter(AbstractDatasetAdapter, StandardEEGChannelsMixin):
    """NeuroGPT dataset adapter."""

    def _setup_adapter(self):
        self.model_name = 'neurogpt'
        self.scale = 0.001  # µV → mV
        super()._setup_adapter()

    def get_supported_channels(self) -> List[str]:
        return self.get_standard_eeg_channels()


class NeuroGPTDataLoaderFactory(AbstractDataLoaderFactory):
    """NeuroGPT DataLoader factory."""

    def create_adapter(
        self,
        dataset: HFDataset,
        dataset_names: List[str],
        dataset_configs: List[str]
    ) -> NeuroGPTDatasetAdapter:
        return NeuroGPTDatasetAdapter(dataset, dataset_names, dataset_configs)