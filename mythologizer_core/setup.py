from mythologizer_core.types import EmbeddingFunction
from mythologizer_core import AgentAttribute
from mythologizer_postgres.db import apply_schemas, drop_everything
from mythologizer_postgres.connectors.mytheme_store import insert_mythemes_bulk
from mythologizer_postgres.connectors import insert_agent_attribute_defs, insert_cultures_bulk
from mythologizer_core.utils import get_embedding_function, validate_embedding_dim


import logging
import numpy as np
from typing import List, Union, Optional, Tuple

logger = logging.getLogger(__name__)





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
    agent_attributes: List[AgentAttribute],
    inital_cultures: Optional[List[Tuple[str, str]]] | str = None
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
        
        # Get and validate embedding function and dimension
        logger.info("Creating SentenceTransformer...")
        embedding_function = get_embedding_function(embedding_function)
        logger.info(f"Created SentenceTransformer: {type(embedding_function)}")
        
        logger.info("Validating embedding dimension...")
        embedding_dim = validate_embedding_dim(embedding_function)
        logger.info(f"Validated embedding dimension: {embedding_dim}")
        
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

        # Process inital cultures
        if inital_cultures:
            if isinstance(inital_cultures, str):
                with open(inital_cultures, "r") as f:
                    inital_cultures = f.readlines()
                inital_cultures = [culture.strip() for culture in inital_cultures]
                inital_cultures = [(culture.split(";")[0], culture.split(";")[1]) for culture in inital_cultures]
                logger.info(f"Loaded inital cultures: {inital_cultures}")

            logger.info("Inserting inital cultures...")
            insert_cultures_bulk(inital_cultures)
            logger.info(f"Successfully inserted {len(inital_cultures)} inital cultures")
        
        logger.info("Simulation setup completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to setup simulation: {str(e)}")
        raise







