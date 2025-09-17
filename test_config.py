#!/usr/bin/env python3
"""
Test script for configuration loading.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from utils.config import ConfigManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_config_loading():
    """Test configuration loading functionality."""
    logger.info("Testing configuration loading...")
    
    config_manager = ConfigManager()
    
    # Test base config loading
    logger.info("Loading base config...")
    base_config = config_manager.load_config("config/base_config.yaml")
    logger.info(f"Base config keys: {list(base_config.keys())}")
    logger.info(f"Model name: {base_config.get('model', {}).get('name')}")
    
    # Test SFT config loading
    logger.info("Loading SFT training config...")
    sft_config = config_manager.load_training_config("sft")
    logger.info(f"SFT config keys: {list(sft_config.keys())}")
    logger.info(f"Batch size: {sft_config.get('training', {}).get('batch_size')}")
    logger.info(f"LoRA rank: {sft_config.get('model', {}).get('lora_config', {}).get('r')}")
    
    # Test config validation
    logger.info("Validating config...")
    is_valid = config_manager.validate_config(sft_config)
    logger.info(f"Config validation: {is_valid}")
    
    logger.info("✓ Configuration loading tests passed")
    return sft_config


def main():
    """Run configuration tests."""
    try:
        config = test_config_loading()
        logger.info("🎉 All configuration tests passed!")
        
        # Show final merged config structure
        logger.info("\nFinal configuration structure:")
        for section, content in config.items():
            if isinstance(content, dict):
                logger.info(f"  {section}: {list(content.keys())}")
            else:
                logger.info(f"  {section}: {content}")
                
    except Exception as e:
        logger.error(f"Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
