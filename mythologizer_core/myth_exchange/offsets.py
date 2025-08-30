from typing import List, Dict, Tuple, Optional
import numpy as np
import logging

from mythologizer_postgres.connectors import get_agents_cultures_ids_bulk
from mythologizer_postgres.connectors.mythicalgebra import get_myth_embeddings
from mythologizer_core.types import Embedding, Embeddings
from mythologizer_core.myth_exchange.utils import average_embeddings

logger = logging.getLogger(__name__)


def get_culture_offsets(
    agent_ids: List[int],
    culture_embedding_dict: Dict[int, Embedding],
    culture_weight: float
) -> List[Embedding]:
    """Calculate culture offsets for both agents."""
    
    try:
        agents_culture_ids = get_agents_cultures_ids_bulk(agent_ids)
        logger.debug(f"Retrieved cultures ids for agents {agent_ids}")
    except Exception as e:
        error_msg = f"Failed to retrieve cultures for agents {agent_ids}: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e

    embeddings = []
    for culture_ids in agents_culture_ids:
        culture_embeddings = []
        for culture_id in culture_ids: 
            if culture_id in culture_embedding_dict:
                culture_embeddings.append(culture_embedding_dict[culture_id])
            else:
                logger.warning(f"Culture id {culture_id} not found in culture_embedding_dict")
        embeddings.append(average_embeddings(culture_embeddings) * culture_weight)

    return embeddings


def calculate_agent_offsets(
    listener_agent_values: List[float] | np.ndarray,
    speaker_agent_values: List[float] | np.ndarray,
    embeddings_of_attribute_names: Embeddings,
    weight_of_attribute_embeddings: float,
    listener_culture_offset: Embedding,
    speaker_culture_offset: Embedding,
    event_embedding: Optional[Embedding],
    event_weight: float
) -> Tuple[Embedding, Embedding]:
    """Calculate the total offsets for both agents."""
    logger.debug(f"Starting _calculate_agent_offsets")
    logger.debug(f"listener_agent_values type: {type(listener_agent_values)}, value: {listener_agent_values}")
    logger.debug(f"speaker_agent_values type: {type(speaker_agent_values)}, value: {speaker_agent_values}")
    logger.debug(f"embeddings_of_attribute_names type: {type(embeddings_of_attribute_names)}, length: {len(embeddings_of_attribute_names)}")
    logger.debug(f"First embedding type: {type(embeddings_of_attribute_names[0]) if embeddings_of_attribute_names else 'None'}")
    
    # Convert agent values to numpy arrays
    listener_values = np.array(listener_agent_values)
    speaker_values = np.array(speaker_agent_values)
    
    # Convert embeddings to numpy array
    embeddings_array = np.array(embeddings_of_attribute_names)
    
    # Calculate weighted attribute offsets by multiplying each agent value with its corresponding embedding
    listener_attribute_offset = np.sum(
        listener_values[:, np.newaxis] * embeddings_array * weight_of_attribute_embeddings, 
        axis=0
    )
    speaker_attribute_offset = np.sum(
        speaker_values[:, np.newaxis] * embeddings_array * weight_of_attribute_embeddings, 
        axis=0
    )
    
    # Handle event offset
    event_offset = event_embedding * event_weight if event_embedding is not None else np.zeros_like(listener_culture_offset)
    
    # Combine all offsets
    listener_offset = listener_attribute_offset + listener_culture_offset + event_offset
    speaker_offset = speaker_attribute_offset + speaker_culture_offset + event_offset
    
    return listener_offset, speaker_offset
