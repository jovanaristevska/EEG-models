"""
NeuroGPT Configuration that inherits from AbstractConfig.
"""
from typing import Dict, Optional, List
from pydantic import Field
from baseline.abstract.config import AbstractConfig, BaseDataArgs, BaseModelArgs, BaseTrainingArgs, BaseLoggingArgs


class NeuroGPTDataArgs(BaseDataArgs):
    """NeuroGPT data configuration."""
    datasets: Dict[str, str] = Field(default_factory=lambda: {})
    batch_size: int = 32
    num_workers: int = 2


class NeuroGPTModelArgs(BaseModelArgs):
    """NeuroGPT model configuration."""
    # Pretrained model path
    pretrained_path: Optional[str] = None

    # EEGConformer (encoder) parameters
    num_encoder_layers: int = 6       # att_depth in EEGConformer
    n_filters_time: int = 40          # embedding dim of encoder output
    filter_time_length: int = 25
    pool_time_length: int = 75
    pool_time_stride: int = 15
    drop_prob: float = 0.5
    att_heads: int = 10
    att_drop_prob: float = 0.5

    # GPT (decoder) parameters
    num_hidden_layers: int = 6
    embedding_dim: int = 1024         # GPT embedding dimension
    num_attention_heads: int = 12
    n_positions: int = 512
    dropout: float = 0.1

    # Input parameters
    chunk_len: int = 500              # window length in samples
    num_chunks: int = 2               # number of chunks
    chunk_ovlp: int = 0               # overlap between chunks

    # Fine-tuning options
    ft_only_encoder: bool = True      # fine-tune only encoder


class NeuroGPTTrainingArgs(BaseTrainingArgs):
    """NeuroGPT training configuration."""
    max_epochs: int = 50
    weight_decay: float = 0.01
    max_grad_norm: float = 3.0
    lr_schedule: str = "cosine"
    max_lr: float = 1e-4
    encoder_lr_scale: float = 1.0
    warmup_epochs: int = 5
    warmup_scale: float = 0.1
    pct_start: float = 0.1
    min_lr: float = 5e-4
    use_amp: bool = False
    freeze_encoder: bool = False
    label_smoothing: float = 0.1


class NeuroGPTLoggingArgs(BaseLoggingArgs):
    """NeuroGPT logging configuration."""
    experiment_name: str = "neurogpt"
    run_dir: str = "assets/run"
    use_cloud: bool = True
    cloud_backend: str = "wandb"
    project: Optional[str] = "neurogpt"
    entity: Optional[str] = None
    api_key: Optional[str] = None
    offline: bool = False
    tags: List[str] = Field(default_factory=lambda: ["neurogpt"])
    log_step_interval: int = 2
    ckpt_interval: int = 10


class NeuroGPTConfig(AbstractConfig):
    """NeuroGPT configuration that extends AbstractConfig."""
    model_type: str = "neurogpt"
    fs: int = 256

    data: NeuroGPTDataArgs = Field(default_factory=NeuroGPTDataArgs)
    model: NeuroGPTModelArgs = Field(default_factory=NeuroGPTModelArgs)
    training: NeuroGPTTrainingArgs = Field(default_factory=NeuroGPTTrainingArgs)
    logging: NeuroGPTLoggingArgs = Field(default_factory=NeuroGPTLoggingArgs)

    def validate_config(self) -> bool:
        if self.model.embedding_dim % self.model.num_attention_heads != 0:
            return False
        if self.training.lr_schedule not in ["onecycle", "cosine"]:
            return False
        return True