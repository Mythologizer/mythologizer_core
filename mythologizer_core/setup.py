from mythologizer_core.types import EmbeddingFunction
from mythologizer_core import AgentAttribute
from mythologizer_postgres.db import apply_schemas, drop_everything
from mythologizer_postgres.connectors.mytheme_store import insert_mythemes_bulk
from mythologizer_postgres.connectors import insert_agent_attribute_defs
from sentence_transformers import SentenceTransformer


import logging
import numpy as np
from typing import List, Union

logger = logging.getLogger(__name__)


def _get_embedding_dim(embedding_function: EmbeddingFunction) -> int:
    """Determine the dimension of embeddings produced by the embedding function."""
    logger.info(f"Testing embedding function with type: {type(embedding_function)}")
    
    # Check if it's a SentenceTransformer and use encode method
    if hasattr(embedding_function, 'encode'):
        logger.info("Using SentenceTransformer.encode() method")
        test_embedding = embedding_function.encode("test")
    else:
        test_embedding = embedding_function("test")
    
    logger.info(f"Test embedding result type: {type(test_embedding)}")
    logger.info(f"Test embedding result: {test_embedding}")
    
    if isinstance(test_embedding, list):
        logger.info(f"Embedding is list with length: {len(test_embedding)}")
        return len(test_embedding)
    elif isinstance(test_embedding, np.ndarray):
        logger.info(f"Embedding is numpy array with shape: {test_embedding.shape}")
        return test_embedding.shape[0]
    else:
        logger.error(f"Unsupported embedding type: {type(test_embedding)}")
        raise ValueError(f"Unsupported embedding type: {type(test_embedding)}")


def _compute_embeddings(embedding_function: EmbeddingFunction, mythemes: List[str]) -> List[Union[List[float], np.ndarray]]:
    """Compute embeddings for all mythemes."""
    logger.info(f"Computing embeddings for {len(mythemes)} mythemes")
    
    # Check if it's a SentenceTransformer and use encode method
    if hasattr(embedding_function, 'encode'):
        logger.info("Using SentenceTransformer.encode() method for batch processing")
        # Use encode method which can handle lists efficiently
        embeddings = embedding_function.encode(mythemes)
        # Convert to list of individual embeddings
        if isinstance(embeddings, np.ndarray):
            return [embeddings[i] for i in range(len(mythemes))]
        else:
            return embeddings
    else:
        return [embedding_function(mytheme) for mytheme in mythemes]


def _validate_inputs(mythemes: List[str], agent_attributes: List[AgentAttribute]) -> None:
    """Validate input parameters."""
    if not mythemes:
        raise ValueError("mythemes list cannot be empty")
    
    if not agent_attributes:
        raise ValueError("agent_attributes list cannot be empty")
    
    logger.info(f"Validated inputs: {len(mythemes)} mythemes, {len(agent_attributes)} agent attributes")


def setup_simulation(
    embedding_function: EmbeddingFunction | str,
    mythemes: List[str] | str,
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
        if isinstance(mythemes, str):
            # read from file
            logger.info(f"Reading mythemes from file: {mythemes}")
            with open(mythemes, "r") as f:
                mythemes = f.readlines()
            mythemes = [mytheme.strip() for mytheme in mythemes]
            logger.info(f"Loaded mythemes: {mythemes}")
            
        _validate_inputs(mythemes, agent_attributes)
        
        # Determine embedding dimension
        logger.info("Creating SentenceTransformer...")
        if isinstance(embedding_function, str):
            embedding_function = SentenceTransformer(embedding_function)
        logger.info(f"Created SentenceTransformer: {type(embedding_function)}")
        
        # Test the embedding function directly
        logger.info("Testing embedding function directly...")
        if hasattr(embedding_function, 'encode'):
            test_result = embedding_function.encode("test")
        else:
            test_result = embedding_function("test")
        logger.info(f"Direct test result type: {type(test_result)}")
        logger.info(f"Direct test result: {test_result}")
        
        logger.info("Getting embedding dimension...")
        embedding_dim = _get_embedding_dim(embedding_function)
        logger.info(f"Detected embedding dimension: {embedding_dim}")
        
        # Reset database
        logger.info("Resetting database...")
        logger.info("Calling drop_everything()...")
        drop_everything()
        logger.info("Calling apply_schemas()...")
        apply_schemas(embedding_dim)
        logger.info("Database reset complete")
        
        # Process mythemes
        logger.info("Computing mytheme embeddings...")
        mytheme_embeddings = _compute_embeddings(embedding_function, mythemes)
        logger.info("Inserting mythemes...")
        insert_mythemes_bulk(mythemes, mytheme_embeddings)
        logger.info(f"Successfully inserted {len(mythemes)} mytheme embeddings")
        
        # Process agent attributes
        logger.info("Inserting agent attributes...")
        insert_agent_attribute_defs(agent_attributes)
        logger.info(f"Successfully inserted {len(agent_attributes)} agent attribute definitions")
        
        logger.info("Simulation setup completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to setup simulation: {str(e)}")
        raise







