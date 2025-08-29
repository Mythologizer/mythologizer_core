from mythologizer_postgres.db import ping_db_basic, check_if_tables_exist
from mythologizer_postgres.connectors import increment_epoch, get_current_epoch
from mythologizer_core.agent_attribute import AgentAttribute
from mythologizer_core.types import EmbeddingFunction
from mythologizer_core.utils import get_embedding_function, validate_embedding_dim


import logging
import numpy as np
from typing import List, Union, Optional
import time


logger = logging.getLogger(__name__)

def _pre_simulation_checks(embedding_function: Union[EmbeddingFunction, str]):
    logger.info("Setting up simulation")
    logger.info("Checking DB connection")
    connection = ping_db_basic()
    if connection is False:
        raise Exception("Failed to connect to DB")
    logger.info("DB connection successful")

    logger.info("Checking tables")
    expected_tables = [ "mythemes", "myths", "agent_attribute_defs", "agent_attributes", "agent_cultures", "agent_myths", "myth_writings", "epoch", "events"]
    table_status = check_if_tables_exist(expected_tables)
    for table, exists in table_status.items():
        if exists is False:
            raise Exception(f"Table {table} does not exist")
    logger.info("All tables exist")
    
    logger.info("Getting and validating embedding function...")
    resolved_embedding_function = get_embedding_function(embedding_function)
    embedding_dim = validate_embedding_dim(resolved_embedding_function)
    logger.info(f"Validated embedding dimension: {embedding_dim}")


def run_simulation(
    agent_attributes: List[AgentAttribute],
    embedding_function: Union[EmbeddingFunction, str],
    n_epochs: Optional[int] = None,
    should_cancel: Optional[callable] = None,
    myth_exchange_config: Optional[dict] = None):
    _pre_simulation_checks(embedding_function)
    
    # Resolve the embedding function once for the entire simulation
    from mythologizer_core.utils.embedding_utils import get_embedding_function
    resolved_embedding_function = get_embedding_function(embedding_function)

    if n_epochs is None:
        logger.info(f"Running simulation for infinite epochs")
    else:
        logger.info(f"Running simulation for {n_epochs} epochs")

    starting_epoch = get_current_epoch()
    current_epoch = starting_epoch
    if n_epochs:
        end_epoch = starting_epoch + n_epochs
    else:
        end_epoch = None
        
    while True:
        # Check for cancellation
        if should_cancel and should_cancel():
            logger.info("Simulation cancelled by user")
            break
            
        logger.info(f"Epoch {current_epoch} starting")
        start_time = time.time()
        try:
            _run_epoch(agent_attributes, resolved_embedding_function, myth_exchange_config)
            epoch_duration = time.time() - start_time
            logger.info(f"Epoch {current_epoch} finished in {epoch_duration:.2f} seconds")
        except Exception as e:
            epoch_duration = time.time() - start_time
            logger.error(f"Epoch {current_epoch} failed after {epoch_duration:.2f} seconds: {str(e)}")
            
            # Check if we should exit on failure
            exit_on_fail = myth_exchange_config.get("exit_on_fail", True) if myth_exchange_config else True
            if exit_on_fail:
                logger.error("Exiting simulation due to epoch failure (exit_on_fail=True)")
                raise
            else:
                logger.warning("Continuing to next epoch despite failure (exit_on_fail=False)")
                # Continue to next epoch
                current_epoch += 1
                if n_epochs is not None and current_epoch > end_epoch:
                    logger.info(f"Epoch {current_epoch-1} reached, stopping simulation")
                    break
                increment_epoch()
                continue
        
        # Get minimum epoch time from configuration (optional)
        if myth_exchange_config and "min_epoch_time" in myth_exchange_config:
            min_epoch_time = myth_exchange_config["min_epoch_time"]
            if min_epoch_time is not None:
                # Calculate sleep time to ensure minimum epoch duration
                sleep_time = max(0, min_epoch_time - epoch_duration)
                if sleep_time > 0:
                    logger.info(f"Waiting {sleep_time:.2f} seconds to meet minimum epoch time of {min_epoch_time} seconds")
                    time.sleep(sleep_time)
            else:
                logger.debug("Minimum epoch time is None, running through without delay")
        else:
            logger.debug("No minimum epoch time specified, running through without delay")
        
        current_epoch += 1
        if n_epochs is not None and current_epoch > end_epoch:
            logger.info(f"Epoch {current_epoch-1} reached, stopping simulation")
            break
        increment_epoch()


def _run_epoch(agent_attributes: List[AgentAttribute], embedding_function: EmbeddingFunction, myth_exchange_config: Optional[dict] = None):
    logger.info("Running epoch")
    
    # Import the run_epoch function
    from mythologizer_core.epoch.epoch import run_epoch
    
    # Get values from configuration or use defaults
    if myth_exchange_config:
        number_of_interactions = myth_exchange_config.get("number_of_interactions", 4)
        max_number_of_listeners = myth_exchange_config.get("max_number_of_listeners", 3)
    else:
        number_of_interactions = 4
        max_number_of_listeners = 3
    
    logger.info(f"Running epoch with {number_of_interactions} interactions, max {max_number_of_listeners} listeners per interaction")
    
    # Run the epoch with myth exchange configuration
    run_epoch(
        agent_attributes=agent_attributes,
        embedding_function=embedding_function,
        number_of_interactions=number_of_interactions,
        max_number_of_listeners=max_number_of_listeners,
        myth_exchange_config=myth_exchange_config
    )







    