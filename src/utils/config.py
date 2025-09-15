"""
Configuration management utilities.
Handles loading, merging, and validating configuration files.
"""

import os
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    """Model configuration settings."""
    name: str
    max_length: int
    temperature: float
    use_peft: bool
    lora_config: Dict[str, Any]


@dataclass
class TrainingConfig:
    """Training configuration settings."""
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    num_epochs: int
    warmup_steps: int
    save_steps: int
    logging_steps: int
    evaluation_steps: int


@dataclass
class RLConfig:
    """Reinforcement learning configuration settings."""
    enabled: bool
    ppo_epochs: int
    mini_batch_size: int
    clip_range: float
    vf_coef: float
    ent_coef: float
    max_grad_norm: float


@dataclass
class CurriculumConfig:
    """Curriculum learning configuration settings."""
    enabled: bool
    difficulty_estimation_method: str
    initial_difficulty: float
    difficulty_increment: float
    max_difficulty: float


@dataclass
class RewardConfig:
    """Reward function configuration settings."""
    length_penalty_weight: float
    max_reasoning_length: int
    correctness_weight: float


@dataclass
class DataConfig:
    """Data configuration settings."""
    datasets: list
    train_split: str
    eval_split: str
    max_samples: int
    preprocessing: Dict[str, Any]


@dataclass
class EvaluationConfig:
    """Evaluation configuration settings."""
    benchmarks: list
    metrics: list
    save_predictions: bool


@dataclass
class HardwareConfig:
    """Hardware and environment configuration settings."""
    device: str
    mixed_precision: bool
    gradient_checkpointing: bool
    dataloader_num_workers: int


@dataclass
class LoggingConfig:
    """Logging and monitoring configuration settings."""
    use_wandb: bool
    project_name: str
    log_dir: str
    save_checkpoints: bool
    checkpoint_dir: str


class ConfigManager:
    """
    Configuration manager for loading and merging configuration files.
    """
    
    def __init__(self, base_config_path: Optional[str] = None):
        """Initialize configuration manager."""
        self.base_config_path = base_config_path or "config/base_config.yaml"
        self.config = {}
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        pass
    
    def merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configuration dictionaries with override priority."""
        pass
    
    def load_training_config(self, training_type: str = "sft") -> Dict[str, Any]:
        """Load training configuration (SFT or RL)."""
        pass
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration for required fields and consistency."""
        pass
    
    def save_config(self, config: Dict[str, Any], output_path: str):
        """Save configuration to file."""
        pass
    
    def get_model_config(self) -> ModelConfig:
        """Get model configuration as dataclass."""
        pass
    
    def get_training_config(self) -> TrainingConfig:
        """Get training configuration as dataclass."""
        pass
    
    def get_rl_config(self) -> RLConfig:
        """Get RL configuration as dataclass."""
        pass
    
    def get_curriculum_config(self) -> CurriculumConfig:
        """Get curriculum configuration as dataclass."""
        pass
    
    def get_reward_config(self) -> RewardConfig:
        """Get reward configuration as dataclass."""
        pass
    
    def get_data_config(self) -> DataConfig:
        """Get data configuration as dataclass."""
        pass
    
    def get_evaluation_config(self) -> EvaluationConfig:
        """Get evaluation configuration as dataclass."""
        pass
    
    def get_hardware_config(self) -> HardwareConfig:
        """Get hardware configuration as dataclass."""
        pass
    
    def get_logging_config(self) -> LoggingConfig:
        """Get logging configuration as dataclass."""
        pass


def load_config_from_args(args) -> Dict[str, Any]:
    """
    Load configuration from command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Configuration dictionary
    """
    pass


def create_config_from_template(template_name: str, output_path: str, **kwargs):
    """
    Create configuration file from template.
    
    Args:
        template_name: Name of configuration template
        output_path: Path to save configuration
        **kwargs: Configuration overrides
    """
    pass


def validate_paths_in_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and resolve paths in configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configuration with validated paths
    """
    pass
