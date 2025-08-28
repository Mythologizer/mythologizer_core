from typing import Tuple
import logging

from mythologizer_postgres.connectors import get_myth_ids_and_retention_from_agents_memory, update_retentions_and_reorder
from mythologizer_postgres.connectors.mythicalgebra import get_myth_embeddings, update_myth_with_retention, insert_myth_to_agent_memory
from mythologizer_core.types import id_list, Embeddings, id_type, Weights, Mythmatrix, Embedding


logger = logging.getLogger(__name__)


def get_agent_myths(agent_id: id_type) -> Tuple[id_list, Weights, Embeddings]:
    """Get myth IDs, retentions, and embeddings for an agent."""
    logger.debug(f"Retrieving myths for agent {agent_id}")
    
    try:
        myth_ids, retentions = get_myth_ids_and_retention_from_agents_memory(agent_id)
        logger.debug(f"Agent {agent_id} has {len(myth_ids)} myths")
    except Exception as e:
        error_msg = f"Failed to retrieve myth IDs and retentions for agent {agent_id}: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e
    
    try:
        myth_embeddings = get_myth_embeddings(myth_ids)
        logger.debug(f"Retrieved embeddings for {len(myth_embeddings)} myths")
    except Exception as e:
        error_msg = f"Failed to retrieve myth embeddings for agent {agent_id}: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e
    
    return myth_ids, retentions, myth_embeddings


def insert_new_myth(
    agent_id: id_type,
    myth_matrix: Mythmatrix,
    mytheme_ids: id_list,
    myth_embedding: Embedding,
) -> None:
    """Insert a new myth into the agent's memory."""
    logger.info(f"Inserting new myth into agent {agent_id}")
    try:
        insert_myth_to_agent_memory(agent_id, myth_matrix, mytheme_ids, myth_embedding)
        logger.info(f"Successfully inserted new myth for agent {agent_id}")
    except Exception as e:
        error_msg = f"Failed to insert new myth for agent {agent_id}: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e

def update_myth_in_listener_memory(
    agent_id: id_type,
    myth_id: id_type,
    myth_matrix: Mythmatrix,
    mytheme_ids: id_list,
    myth_embedding: Embedding,
    retention: float):
    """Update an existing myth in the agent's memory."""
    logger.info(f"Updating myth {myth_id} in agent {agent_id}")
    try:
        update_myth_with_retention(agent_id, myth_id, myth_matrix, mytheme_ids, retention, myth_embedding)
        logger.info(f"Successfully updated myth {myth_id} in agent {agent_id}")
    except Exception as e:
        error_msg = f"Failed to update myth {myth_id} in agent {agent_id}: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def update_speaker_retention(
    speaker_agent_id: id_type,
    myth_ids: id_list,
    retentions: Weights,
    chosen_myth_index: int,
    retention_remember_factor: float,
    retention_forget_factor: float
) -> None:
    """Update the speaker's myth retention values."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Updating speaker {speaker_agent_id} retention values")
    logger.debug(f"Original retentions: {retentions}")
    logger.debug(f"Chosen myth index: {chosen_myth_index}")
    logger.debug(f"Remember factor: {retention_remember_factor}, Forget factor: {retention_forget_factor}")
    
    # Apply forget factor to all myths except the chosen one
    for i in range(len(retentions)):
        if i != chosen_myth_index:
            old_retention = retentions[i]
            retentions[i] = max(0.0, old_retention - retention_forget_factor)
            logger.debug(f"Myth {i} (ID: {myth_ids[i]}): {old_retention} -> {retentions[i]} (forgotten)")
        else:
            old_retention = retentions[i]
            retentions[i] += retention_remember_factor
            logger.debug(f"Myth {i} (ID: {myth_ids[i]}): {old_retention} -> {retentions[i]} (remembered)")
    
    logger.debug(f"Updated retentions: {retentions}")
    
    myth_id_retention_tuples = [
        (myth_id, retention) 
        for myth_id, retention in zip(myth_ids, retentions)
    ]
    logger.info(f"Updating database with {len(myth_id_retention_tuples)} myth retention tuples")
    
    try:
        update_retentions_and_reorder(speaker_agent_id, myth_id_retention_tuples)
        logger.info(f"Successfully updated speaker {speaker_agent_id} retention values")
    except Exception as e:
        error_msg = f"Failed to update retention values for speaker {speaker_agent_id}: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e