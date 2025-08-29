#!/usr/bin/env python3
"""
Test epoch functionality step by step.
This file replicates the same setup as the GUI for testing the epoch function.
"""

import logging
import asyncio
import importlib.util
import os
import json
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_epoch.log')
    ]
)

logger = logging.getLogger(__name__)

# Configuration file handling
CONFIG_FILE = Path("simulation_config.json")


def load_simulation_config() -> Dict[str, Any]:
    """Load simulation configuration from JSON file."""
    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
    
    # Return default configuration
    return {
        "embedding_model": "all-MiniLM-L6-v2",
        "agent_attributes_file": "agent_attributes.py",
        "mythemes_file": "mythemes.txt",
        "initial_cultures_file": "",
        "epochs": "4"
    }


def load_agent_attributes(config: Dict[str, Any]) -> List:
    """Load agent attributes from the specified file, same as GUI."""
    logger.info(f"Loading agent attributes from: {config['agent_attributes_file']}")
    
    # Check if agent attributes file exists
    agent_attributes_file = config["agent_attributes_file"]
    if not os.path.exists(agent_attributes_file):
        raise FileNotFoundError(f"Agent attributes file not found: {agent_attributes_file}")
    
    logger.info(f"Agent attributes file exists: {agent_attributes_file}")
    
    # Load the agent attributes file
    spec = importlib.util.spec_from_file_location("agent_attributes_module", agent_attributes_file)
    agent_attributes_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent_attributes_module)
    
    # Get the agent_attributes list from the module
    agent_attributes = getattr(agent_attributes_module, "agent_attributes")
    logger.info(f"Loaded {len(agent_attributes)} agent attributes")
    logger.debug(f"Agent attributes: {agent_attributes}")
    
    return agent_attributes


def get_embedding_function(config: Dict[str, Any]):
    """Get embedding function from config, same as GUI."""
    logger.info(f"Creating embedding function from model: {config['embedding_model']}")
    
    try:
        from mythologizer_core.utils import get_embedding_function
        embedding_function = get_embedding_function(config["embedding_model"])
        logger.info(f"Successfully created embedding function: {type(embedding_function)}")
        return embedding_function
    except Exception as e:
        logger.error(f"Failed to create embedding function: {str(e)}")
        raise


def test_step_1_load_config():
    """Test step 1: Load configuration."""
    logger.info("=" * 50)
    logger.info("STEP 1: Loading configuration")
    logger.info("=" * 50)
    
    try:
        config = load_simulation_config()
        logger.info(f"Configuration loaded successfully: {config}")
        return config
    except Exception as e:
        logger.error(f"Step 1 failed: {str(e)}")
        raise


def test_step_2_load_agent_attributes(config: Dict[str, Any]):
    """Test step 2: Load agent attributes."""
    logger.info("=" * 50)
    logger.info("STEP 2: Loading agent attributes")
    logger.info("=" * 50)
    
    try:
        agent_attributes = load_agent_attributes(config)
        logger.info(f"Agent attributes loaded successfully: {len(agent_attributes)} attributes")
        return agent_attributes
    except Exception as e:
        logger.error(f"Step 2 failed: {str(e)}")
        raise


def test_step_3_create_embedding_function(config: Dict[str, Any]):
    """Test step 3: Create embedding function."""
    logger.info("=" * 50)
    logger.info("STEP 3: Creating embedding function")
    logger.info("=" * 50)
    
    try:
        embedding_function = get_embedding_function(config)
        logger.info(f"Embedding function created successfully: {type(embedding_function)}")
        return embedding_function
    except Exception as e:
        logger.error(f"Step 3 failed: {str(e)}")
        raise


def test_step_4_validate_embedding_function(embedding_function):
    """Test step 4: Validate embedding function."""
    logger.info("=" * 50)
    logger.info("STEP 4: Validating embedding function")
    logger.info("=" * 50)
    
    try:
        from mythologizer_core.utils import validate_embedding_dim
        
        # Test the embedding function
        test_text = "test"
        logger.info(f"Testing embedding function with text: '{test_text}'")
        
        # Use the encode method for SentenceTransformer
        if hasattr(embedding_function, 'encode'):
            embedding = embedding_function.encode(test_text)
        else:
            embedding = embedding_function(test_text)
        
        logger.info(f"Test embedding created successfully: {type(embedding)}, shape: {embedding.shape if hasattr(embedding, 'shape') else len(embedding)}")
        
        # Validate embedding dimension
        embedding_dim = validate_embedding_dim(embedding_function)
        logger.info(f"Embedding dimension validated: {embedding_dim}")
        
        return embedding_dim
    except Exception as e:
        logger.error(f"Step 4 failed: {str(e)}")
        raise


def test_step_5_test_epoch_function(agent_attributes, embedding_function, config: Dict[str, Any]):
    """Test step 5: Test the epoch function."""
    logger.info("=" * 50)
    logger.info("STEP 5: Testing epoch function")
    logger.info("=" * 50)
    
    try:
        from mythologizer_core.epoch.epoch import run_epoch
        
        # Get number of interactions from config or use default
        number_of_interactions = int(config.get("epochs", 4))
        max_number_of_listeners = 3  # Default value
        
        logger.info(f"Running epoch with parameters:")
        logger.info(f"  - Number of interactions: {number_of_interactions}")
        logger.info(f"  - Max listeners per interaction: {max_number_of_listeners}")
        logger.info(f"  - Number of agent attributes: {len(agent_attributes)}")
        
        # Run the epoch
        run_epoch(
            agent_attributes=agent_attributes,
            embedding_function=embedding_function,
            number_of_interactions=number_of_interactions,
            max_number_of_listeners=max_number_of_listeners
        )
        
        logger.info("Epoch completed successfully!")
        return True
    except Exception as e:
        logger.error(f"Step 5 failed: {str(e)}")
        raise


def test_step_6_run_multiple_epochs(agent_attributes, embedding_function, config: Dict[str, Any]):
    """Test step 6: Run multiple epochs."""
    logger.info("=" * 50)
    logger.info("STEP 6: Running multiple epochs")
    logger.info("=" * 50)
    
    try:
        from mythologizer_core.epoch.epoch import run_epoch
        
        number_of_epochs = int(config.get("epochs", 4))
        number_of_interactions = 2  # Fewer interactions per epoch for testing
        max_number_of_listeners = 2
        
        logger.info(f"Running {number_of_epochs} epochs with {number_of_interactions} interactions each")
        
        for epoch_num in range(1, number_of_epochs + 1):
            logger.info(f"Starting epoch {epoch_num}/{number_of_epochs}")
            
            run_epoch(
                agent_attributes=agent_attributes,
                embedding_function=embedding_function,
                number_of_interactions=number_of_interactions,
                max_number_of_listeners=max_number_of_listeners
            )
            
            logger.info(f"Completed epoch {epoch_num}/{number_of_epochs}")
            
            # Small delay between epochs
            import time
            time.sleep(1)
        
        logger.info("All epochs completed successfully!")
        return True
    except Exception as e:
        logger.error(f"Step 6 failed: {str(e)}")
        raise


def run_all_tests():
    """Run all test steps in sequence."""
    logger.info("Starting comprehensive epoch testing")
    logger.info("=" * 80)
    
    try:
        # Step 1: Load configuration
        config = test_step_1_load_config()
        
        # Step 2: Load agent attributes
        agent_attributes = test_step_2_load_agent_attributes(config)
        
        # Step 3: Create embedding function
        embedding_function = test_step_3_create_embedding_function(config)
        
        # Step 4: Validate embedding function
        embedding_dim = test_step_4_validate_embedding_function(embedding_function)
        
        # Step 5: Test single epoch
        test_step_5_test_epoch_function(agent_attributes, embedding_function, config)
        
        # Step 6: Test multiple epochs
        test_step_6_run_multiple_epochs(agent_attributes, embedding_function, config)
        
        logger.info("=" * 80)
        logger.info("ALL TESTS PASSED! Epoch functionality is working correctly.")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"TESTING FAILED: {str(e)}")
        logger.error("=" * 80)
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise


def run_single_test(test_name: str):
    """Run a single test step."""
    logger.info(f"Running single test: {test_name}")
    
    try:
        config = load_simulation_config()
        agent_attributes = load_agent_attributes(config)
        embedding_function = get_embedding_function(config)
        
        if test_name == "config":
            test_step_1_load_config()
        elif test_name == "agent_attributes":
            test_step_2_load_agent_attributes(config)
        elif test_name == "embedding_function":
            test_step_3_create_embedding_function(config)
        elif test_name == "validate_embedding":
            test_step_4_validate_embedding_function(embedding_function)
        elif test_name == "single_epoch":
            test_step_5_test_epoch_function(agent_attributes, embedding_function, config)
        elif test_name == "multiple_epochs":
            test_step_6_run_multiple_epochs(agent_attributes, embedding_function, config)
        else:
            logger.error(f"Unknown test: {test_name}")
            logger.info("Available tests: config, agent_attributes, embedding_function, validate_embedding, single_epoch, multiple_epochs")
            return
        
        logger.info(f"Test '{test_name}' completed successfully!")
        
    except Exception as e:
        logger.error(f"Test '{test_name}' failed: {str(e)}")
        raise


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Run specific test
        test_name = sys.argv[1]
        run_single_test(test_name)
    else:
        # Run all tests
        run_all_tests()
