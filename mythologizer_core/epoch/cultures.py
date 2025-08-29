from mythologizer_postgres.connectors import get_all_cultures
from mythologizer_core.types import id_type
from mythologizer_core.types.types import Embedding, EmbeddingFunction

import logging
from typing import Dict, List, Tuple, Union


logger = logging.getLogger(__name__)


def get_all_cultures_safe() -> List[Tuple[id_type, str, str]]:
    """
    Safely retrieve all cultures with error handling.
    
    Returns:
        List of culture tuples (id, name, description)
        
    Raises:
        RuntimeError: If database operation fails
    """
    logger.debug("Retrieving all cultures from database...")
    
    try:
        cultures = get_all_cultures()
        logger.info(f"Successfully retrieved {len(cultures)} cultures")
        logger.debug(f"Culture IDs: {[culture[0] for culture in cultures]}")
        return cultures
    except Exception as e:
        error_msg = f"Failed to retrieve cultures: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e

def create_culture_embedding_dict(
    cultures: List[Tuple[id_type, str, str]],
    embedding_function: Union[EmbeddingFunction, str]
) -> Dict[id_type, Embedding]:
    """
    Create a dictionary mapping culture IDs to their embeddings.
    
    Args:
        cultures: List of culture tuples (id, name, description)
        embedding_function: Function to convert text to embeddings
        
    Returns:
        Dictionary mapping culture ID to embedding
    """
    logger.debug("Creating culture embedding dictionary...")
    
    culture_embedding_dict = {}
    for culture in cultures:
        try:
            culture_id, culture_name, culture_description = culture
            # Use the encode method for SentenceTransformer
            if hasattr(embedding_function, 'encode'):
                embedding = embedding_function.encode(culture_description)
            else:
                embedding = embedding_function(culture_description)
            culture_embedding_dict[culture_id] = embedding
            logger.debug(f"Created embedding for culture {culture_id} ({culture_name})")
        except Exception as e:
            logger.warning(f"Failed to create embedding for culture {culture[0]}: {str(e)}")
            # Continue with other cultures even if one fails
    
    logger.info(f"Created embeddings for {len(culture_embedding_dict)} cultures")
    return culture_embedding_dict