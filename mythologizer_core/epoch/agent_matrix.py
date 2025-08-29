from mythologizer_postgres.connectors import get_agent_attribute_matrix, update_agent_attribute_matrix
from mythologizer_core.agent_attribute import AgentAttribute
from mythologizer_core.types import AgentAttributeMatrix, id_list

import logging
import numpy as np
from typing import List, Union, Optional, Tuple, Dict
import time


logger = logging.getLogger(__name__)


def get_agent_attribute_matrix_safe() -> Tuple[AgentAttributeMatrix, id_list, Dict[str, int]]:
    """
    Safely retrieve the agent attribute matrix with error handling.
    
    Returns:
        Tuple of (attribute_matrix, agent_indices, attribute_name_to_col)
        
    Raises:
        RuntimeError: If database operation fails
    """
    logger.debug("Retrieving agent attribute matrix from database...")
    
    try:
        attribute_matrix, agent_indices, attribute_name_to_col = get_agent_attribute_matrix()
        logger.info(f"Successfully retrieved agent attribute matrix with {len(agent_indices)} agents and {len(attribute_name_to_col)} attributes")
        logger.debug(f"Matrix shape: {attribute_matrix.shape}")
        logger.debug(f"Agent indices: {agent_indices}")
        logger.debug(f"Attribute names: {list(attribute_name_to_col.keys())}")
        return attribute_matrix, agent_indices, attribute_name_to_col
    except Exception as e:
        error_msg = f"Failed to retrieve agent attribute matrix: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def update_agent_attribute_matrix_safe(attribute_matrix: AgentAttributeMatrix, agent_indices: id_list) -> None:
    """
    Safely update the agent attribute matrix with error handling.
    
    Args:
        attribute_matrix: The updated attribute matrix
        agent_indices: List of agent indices
        
    Raises:
        RuntimeError: If database operation fails
    """
    logger.debug("Updating agent attribute matrix in database...")
    
    try:
        update_agent_attribute_matrix(attribute_matrix, agent_indices)
        logger.info(f"Successfully updated agent attribute matrix for {len(agent_indices)} agents")
    except Exception as e:
        error_msg = f"Failed to update agent attribute matrix: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e



def get_and_update_epoch_agent_attributes(agent_attributes: List[AgentAttribute]) -> Tuple[AgentAttributeMatrix, id_list]:
    """
    Apply epoch functions to agent attributes and update the database.
    
    Args:
        agent_attributes: List of agent attributes to process
        
    Returns:
        Tuple of (updated_attribute_matrix, agent_indices)
        
    Raises:
        RuntimeError: If database operations fail
    """
    logger.info("Starting agent attribute epoch processing")
    logger.debug(f"Processing {len(agent_attributes)} agent attributes")
    
    # Get current agent attribute matrix
    attribute_matrix, agent_indices, attribute_name_to_col = get_agent_attribute_matrix_safe()
    
    # Apply epoch functions to attributes that have them
    attributes_with_epoch_functions = [attr for attr in agent_attributes if attr.epoch_change_function]
    logger.info(f"Found {len(attributes_with_epoch_functions)} attributes with epoch functions")
    
    for index, agent_attribute in enumerate(agent_attributes):
        if agent_attribute.epoch_change_function:
            logger.debug(f"Processing attribute: {agent_attribute.name}")
            
            try:
                col_index = attribute_name_to_col[agent_attribute.name]
                logger.debug(f"Attribute '{agent_attribute.name}' found at column index {col_index}")
                
                # Apply the epoch function to the specific column
                old_values = attribute_matrix[:, col_index].copy()
                attribute_matrix[:, col_index] = agent_attribute.epoch_change_function(attribute_matrix[:, col_index])
                
                # Apply min/max constraints if defined
                if agent_attribute.min is not None:
                    attribute_matrix[:, col_index] = np.maximum(attribute_matrix[:, col_index], agent_attribute.min)
                    logger.debug(f"Applied min constraint {agent_attribute.min} to {agent_attribute.name}")
                
                if agent_attribute.max is not None:
                    attribute_matrix[:, col_index] = np.minimum(attribute_matrix[:, col_index], agent_attribute.max)
                    logger.debug(f"Applied max constraint {agent_attribute.max} to {agent_attribute.name}")
                
                new_values = attribute_matrix[:, col_index]
                
                logger.info(f"Epoch function applied to '{agent_attribute.name}'")
                logger.debug(f"Values changed for {agent_attribute.name}: {old_values[:5]} -> {new_values[:5]}")
                
            except KeyError as e:
                error_msg = f"Attribute '{agent_attribute.name}' not found in attribute matrix columns"
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e
            except Exception as e:
                error_msg = f"Failed to apply epoch function to attribute '{agent_attribute.name}': {str(e)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e
    
    # Update the database with the modified matrix
    update_agent_attribute_matrix_safe(attribute_matrix, agent_indices)
    
    logger.info("Agent attribute epoch processing completed successfully")
    return attribute_matrix, agent_indices

