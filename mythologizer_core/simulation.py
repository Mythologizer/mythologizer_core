from mythologizer_postgres.db import ping_db_basic, check_if_tables_exist
from mythologizer_postgres.connectors import increment_epoch, get_current_epoch
from mythologizer_core.agent_attribute import AgentAttribute

import logging
import numpy as np
from typing import List, Union, Optional
import time


logger = logging.getLogger(__name__)

def _pre_simulation_checks():
    logger.info("Setting up simulation")
    logger.info("Checking DB connection")
    connection = ping_db_basic()
    if connection is False:
        raise Exception("Failed to connect to DB")
    logger.info("DB connection successful")

    logger.info("Checking tables")
    expected_tables = [ "mythemes", "myths", "agent_attribute_defs", "agent_attributes", "agent_cultures", "agent_myths", "myth_writings", "epoch"]
    table_status = check_if_tables_exist(expected_tables)
    for table, exists in table_status.items():
        if exists is False:
            raise Exception(f"Table {table} does not exist")
    logger.info("All tables exist")


def run_simulation(
    agent_attributes: List[AgentAttribute],
    n_epochs: Optional[int] = None,
    should_cancel: Optional[callable] = None):
    _pre_simulation_checks()

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







    