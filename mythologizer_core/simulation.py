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
    should_cancel: Optional[callable] = None):
    _pre_simulation_checks(embedding_function)

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
        _run_epoch()
        logger.info(f"Epoch {current_epoch} finished")
        current_epoch += 1
        if n_epochs is not None and current_epoch > end_epoch:
            logger.info(f"Epoch {current_epoch-1} reached, stopping simulation")
            break
        increment_epoch()
        time.sleep(3)


def _run_epoch():
    logger.info("Running epoch")







    