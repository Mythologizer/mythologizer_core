"""
Utility functions for handling embedding functions and related operations.
"""

import logging
import numpy as np
from typing import Union
from sentence_transformers import SentenceTransformer
from mythologizer_core.types import Embedding, EmbeddingFunction
from mythologizer_postgres.db import is_correct_embedding_size

logger = logging.getLogger(__name__)

def get_embedding_from_function(embedding_function: EmbeddingFunction, text: str) -> Embedding:
    """
    Get an embedding from an embedding function.
    
    Args:
        embedding_function: The embedding function to use
        text: The text to embed
    """
    try:
        if hasattr(embedding_function, 'encode'):
            embedding = embedding_function.encode(text)
        else:
            embedding = embedding_function(text)
        return embedding
    except Exception as e:
        error_msg = f"Failed to create embedding for text '{text}': {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def get_embedding_function(embedding_function: Union[EmbeddingFunction, str]) -> EmbeddingFunction:
    """
    Get an embedding function from either a function or a string identifier.
    
    Args:
        embedding_function: Either an embedding function or a string identifier for a SentenceTransformer model
        
    Returns:
        EmbeddingFunction: The resolved embedding function
        
    Raises:
        ValueError: If the embedding function cannot be resolved
    """
    if isinstance(embedding_function, str):
        logger.info(f"Creating SentenceTransformer from string: {embedding_function}")
        try:
            return SentenceTransformer(embedding_function)
        except Exception as e:
            logger.error(f"Failed to create SentenceTransformer from '{embedding_function}': {str(e)}")
            raise ValueError(f"Invalid embedding function string '{embedding_function}': {str(e)}")
    
    logger.info(f"Using provided embedding function: {type(embedding_function)}")
    return embedding_function


def validate_embedding_dim(embedding_function: EmbeddingFunction) -> int:
    """
    Validate that the embedding function produces embeddings with the correct dimension for the database schema.
    
    Args:
        embedding_function: The embedding function to validate
        
    Returns:
        int: The validated embedding dimension
        
    Raises:
        ValueError: If the embedding type is not supported or dimension doesn't match database schema
    """
    logger.info(f"Validating embedding function with type: {type(embedding_function)}")
    
    # Check if it's a SentenceTransformer and use encode method
    if hasattr(embedding_function, 'encode'):
        logger.info("Using SentenceTransformer.encode() method")
        test_embedding = embedding_function.encode("test")
    else:
        test_embedding = embedding_function("test")
    
    logger.info(f"Test embedding result type: {type(test_embedding)}")
    logger.info(f"Test embedding result: {test_embedding}")
    
    # Determine embedding dimension
    if isinstance(test_embedding, list):
        logger.info(f"Embedding is list with length: {len(test_embedding)}")
        embedding_dim = len(test_embedding)
    elif isinstance(test_embedding, np.ndarray):
        logger.info(f"Embedding is numpy array with shape: {test_embedding.shape}")
        embedding_dim = test_embedding.shape[0]
    else:
        logger.error(f"Unsupported embedding type: {type(test_embedding)}")
        raise ValueError(f"Unsupported embedding type: {type(test_embedding)}")
    
    # Validate embedding dimension against database schema
    logger.info(f"Validating embedding dimension {embedding_dim} against database schema...")
    if not is_correct_embedding_size(embedding_dim):
        error_msg = f"Embedding dimension {embedding_dim} does not match the database schema"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info(f"Embedding dimension {embedding_dim} is valid for the database schema")
    return embedding_dim
