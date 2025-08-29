from mythologizer_postgres.connectors import get_agent_attribute_matrix, update_agent_attribute_matrix, get_all_cultures
from mythologizer_core.agent_attribute import AgentAttribute
from mythologizer_core.types import EmbeddingFunction
from mythologizer_core.myth_exchange.myth_exchange import tell_myth
from mythologizer_core.epoch.cultures import get_all_cultures_safe, create_culture_embedding_dict
from mythologizer_core.epoch.agent_matrix import get_and_update_epoch_agent_attributes
from mythologizer_core.epoch.interaction import get_interactions, process_interaction
from mythologizer_core.utils.embedding_utils import get_embedding_from_function


import logging
import numpy as np
from typing import List, Union, Optional, Tuple, Dict
import time


logger = logging.getLogger(__name__)

def run_epoch(
    agent_attributes: List[AgentAttribute],
    embedding_function: Union[EmbeddingFunction, str],
    number_of_interactions: int,
    max_number_of_listeners: int,
    myth_exchange_config: Optional[Dict[str, float]] = None
) -> None:
    """
    Run a complete epoch with agent attribute updates and myth exchanges.
    
    Args:
        agent_attributes: List of agent attributes to process
        embedding_function: Function to convert text to embeddings
        number_of_interactions: Number of interactions to generate
        max_number_of_listeners: Maximum number of listeners per interaction
        
    Raises:
        RuntimeError: If any operation fails
    """
    logger.info("Starting epoch execution")
    logger.info(f"Parameters: {len(agent_attributes)} attributes, {number_of_interactions} interactions, max {max_number_of_listeners} listeners")
    
    try:
        logger.info("Creating attribute name embeddings...")
        embeddings_of_attribute_names = []
        for attribute in agent_attributes:
            embedding = get_embedding_from_function(embedding_function, attribute.name)
            embeddings_of_attribute_names.append(embedding)
        
        logger.info(f"Created embeddings for {len(embeddings_of_attribute_names)} attributes")
        
        logger.info("Processing agent attributes...")
        attribute_matrix, agent_indices = get_and_update_epoch_agent_attributes(agent_attributes)
        
        logger.info("Retrieving cultures and creating embeddings...")
        cultures = get_all_cultures_safe()
        culture_embedding_dict = create_culture_embedding_dict(cultures, embedding_function)
        
        logger.info("Generating interactions...")
        interactions = get_interactions(agent_indices, number_of_interactions, max_number_of_listeners)
        
        # Process each interaction
        logger.info(f"Processing {len(interactions)} interactions...")
        successful_interactions = 0
        
        for i, interaction in enumerate(interactions):
            try:
                logger.info(f"Processing interaction {i+1}/{len(interactions)}: {interaction}")
                process_interaction(
                    interaction,
                    attribute_matrix,
                    agent_indices,
                    embeddings_of_attribute_names,
                    embedding_function,
                    culture_embedding_dict,
                    myth_exchange_config
                )
                successful_interactions += 1
                logger.info(f"Successfully processed interaction {i+1}")
                
            except Exception as e:
                logger.error(f"Failed to process interaction {i+1}: {str(e)}")
                #TODO: Handle this better let n interactions fail
                raise RuntimeError(f"Failed to process interaction {i+1}: {str(e)}") from e

        
        logger.info(f"Epoch completed: {successful_interactions}/{len(interactions)} interactions successful")
        
        if successful_interactions == 0:
            error_msg = f"All {len(interactions)} interactions failed during epoch execution"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
    except Exception as e:
        error_msg = f"Epoch execution failed: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e

