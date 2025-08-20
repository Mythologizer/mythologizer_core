from mythologizer_core.types import EmbeddingFunction
from mythologizer_core import AgentAttribute
from mythologizer_postgres.db import apply_schemas, drop_everything
from mythologizer_postgres.connectors.mytheme_store import insert_mythemes_bulk
from mythologizer_postgres.connectors import insert_agent_attribute_defs

import logging
import numpy as np
from typing import List, Union

logger = logging.getLogger(__name__)


def _get_embedding_dim(embedding_function: EmbeddingFunction) -> int:
    """Determine the dimension of embeddings produced by the embedding function."""
    test_embedding = embedding_function("test")
    
    if isinstance(test_embedding, list):
        return len(test_embedding)
    elif isinstance(test_embedding, np.ndarray):
        return test_embedding.shape[0]
    else:
        raise ValueError(f"Unsupported embedding type: {type(test_embedding)}")


def _compute_embeddings(embedding_function: EmbeddingFunction, mythemes: List[str]) -> List[Union[List[float], np.ndarray]]:
    """Compute embeddings for all mythemes."""
    logger.info(f"Computing embeddings for {len(mythemes)} mythemes")
    return [embedding_function(mytheme) for mytheme in mythemes]


def _validate_inputs(mythemes: List[str], agent_attributes: List[AgentAttribute]) -> None:
    """Validate input parameters."""
    if not mythemes:
        raise ValueError("mythemes list cannot be empty")
    
    if not agent_attributes:
        raise ValueError("agent_attributes list cannot be empty")
    
    logger.info(f"Validated inputs: {len(mythemes)} mythemes, {len(agent_attributes)} agent attributes")


def setup_simulation(
    embedding_function: EmbeddingFunction, 
    mythemes: List[str], 
    agent_attributes: List[AgentAttribute]
) -> None:
    """
    Set up the simulation environment with embeddings and agent attributes.
    
    Args:
        embedding_function: Function to generate embeddings from text
        mythemes: List of mytheme strings to embed
        agent_attributes: List of agent attribute definitions
    
    Raises:
        ValueError: If inputs are invalid or embedding function fails
        Exception: If database operations fail
    """
    try:
        # Validate inputs
        _validate_inputs(mythemes, agent_attributes)
        
        # Determine embedding dimension
        embedding_dim = _get_embedding_dim(embedding_function)
        logger.info(f"Detected embedding dimension: {embedding_dim}")
        
        # Reset database
        logger.info("Resetting database...")
        drop_everything()
        apply_schemas(embedding_dim)
        logger.info("Database reset complete")
        
        # Process mythemes
        mytheme_embeddings = _compute_embeddings(embedding_function, mythemes)
        insert_mythemes_bulk(mythemes, mytheme_embeddings)
        logger.info(f"Successfully inserted {len(mythemes)} mytheme embeddings")
        
        # Process agent attributes
        insert_agent_attribute_defs(agent_attributes)
        logger.info(f"Successfully inserted {len(agent_attributes)} agent attribute definitions")
        
        logger.info("Simulation setup completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to setup simulation: {str(e)}")
        raise







