#!/usr/bin/env python3

import logging
import json
import os
import importlib.util
import asyncio
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# Load environment variables (same as app)
load_dotenv(find_dotenv())

# Import Textual components to replicate the environment
try:
    from textual.logging import TextualHandler
    from textual import work
    from mythologizer_postgres.db import ping_db_basic
    from mythologizer_postgres.connectors import get_simulation_status
    TEXTUAL_AVAILABLE = True
except ImportError:
    TEXTUAL_AVAILABLE = False
    print("Textual not available, using basic logging")

# Configure logging to use TextualHandler (same as app)
if TEXTUAL_AVAILABLE:
    logging.basicConfig(
        level="DEBUG",
        handlers=[TextualHandler()],
    )
else:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

# Replicate the database monitoring from the UI
async def simulate_db_monitoring():
    """Simulate the database monitoring that runs in the UI background."""
    if not TEXTUAL_AVAILABLE:
        return
    
    logger.info("Starting simulated database monitoring...")
    while True:
        try:
            # Simulate the database ping that happens in the UI
            connection_status = await asyncio.to_thread(ping_db_basic)
            logger.debug(f"DB monitoring ping result: {connection_status}")
            
            if connection_status:
                # Simulate getting simulation status
                status = await asyncio.to_thread(get_simulation_status)
                logger.debug(f"DB monitoring status: {status}")
            
        except Exception as e:
            logger.debug(f"DB monitoring error: {e}")
        
        await asyncio.sleep(1)

def load_simulation_config():
    """Load simulation configuration from JSON file."""
    config_file = Path("simulation_config.json")
    try:
        if config_file.exists():
            with open(config_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
    
    # Return default configuration
    return {
        "embedding_model": "all-MiniLM-L6-v2",
        "agent_attributes_file": "agent_attributes.py",
        "mythemes_file": "mythemes.txt"
    }

async def test_setup_simulation():
    """Test the setup simulation with the same parameters as the UI."""
    try:
        logger.info("Starting standalone setup simulation test...")
        
        # Start database monitoring in background (replicating UI environment)
        db_monitoring_task = None
        if TEXTUAL_AVAILABLE:
            db_monitoring_task = asyncio.create_task(simulate_db_monitoring())
            logger.info("Started background database monitoring")
        
        # Wait a bit for monitoring to start
        await asyncio.sleep(2)
        
        # Load the saved configuration
        config = load_simulation_config()
        logger.info(f"Loaded configuration: {config}")
        
        # Import the setup function
        from mythologizer_core.setup import setup_simulation
        logger.info("Imported setup_simulation function")
        
        # Load agent attributes from the specified file
        logger.info(f"Loading agent attributes from: {config['agent_attributes_file']}")
        
        # Check if agent attributes file exists
        agent_attributes_file = config["agent_attributes_file"]
        if not os.path.exists(agent_attributes_file):
            raise FileNotFoundError(f"Agent attributes file not found: {agent_attributes_file}")
        
        # Check if mythemes file exists
        mythemes_file = config["mythemes_file"]
        if not os.path.exists(mythemes_file):
            raise FileNotFoundError(f"Mythemes file not found: {mythemes_file}")
        
        logger.info(f"Files exist: {agent_attributes_file} and {mythemes_file}")
        
        # Load the agent attributes file
        spec = importlib.util.spec_from_file_location("agent_attributes_module", agent_attributes_file)
        agent_attributes_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(agent_attributes_module)
        
        # Get the agent_attributes list from the module
        agent_attributes = getattr(agent_attributes_module, "agent_attributes")
        logger.info(f"Loaded {len(agent_attributes)} agent attributes")
        logger.info(f"Agent attributes type: {type(agent_attributes)}")
        logger.info(f"Agent attributes: {agent_attributes}")
        
        # Call setup_simulation with the saved configuration
        logger.info("Calling setup_simulation...")
        logger.info(f"Parameters: embedding_model={config['embedding_model']}, mythemes_file={config['mythemes_file']}, agent_attributes_count={len(agent_attributes)}")
        
        # Call setup_simulation using asyncio.to_thread (same as UI)
        result = await asyncio.to_thread(
            setup_simulation,
            embedding_function=config["embedding_model"],
            mythemes=config["mythemes_file"],
            agent_attributes=agent_attributes
        )
        
        logger.info("Setup simulation completed successfully!")
        
        # Cancel the database monitoring task
        if db_monitoring_task:
            db_monitoring_task.cancel()
            try:
                await db_monitoring_task
            except asyncio.CancelledError:
                pass
        
    except Exception as e:
        logger.error(f"Setup failed: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise e

if __name__ == "__main__":
    asyncio.run(test_setup_simulation())
