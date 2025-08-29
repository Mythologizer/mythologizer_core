from typing import Callable, Optional, Tuple
import logging
import numpy as np
from mythologizer_core.types import Embedding, Embeddings

logger = logging.getLogger(__name__)


def select_speaker_myth(
    myth_embeddings: Embeddings,
    event_embedding: Optional[Embedding],
    myth_index_sample_function: Callable[[int], int],
    distance_function: Callable[[Embedding, Embedding], float]
) -> int:
    """Select the most appropriate myth from the speaker's memory."""
    
    if event_embedding is not None:
        logger.debug("Selecting myth based on event similarity")
        event_embedding = np.array(event_embedding)
        distances_to_event = [
            distance_function(myth_embedding, event_embedding) 
            for myth_embedding in myth_embeddings
        ]
        chosen_index = np.argmin(distances_to_event)
        logger.debug(f"Selected myth at index {chosen_index} with distance {distances_to_event[chosen_index]}")
        return chosen_index
    else:
        logger.debug("No event provided, using sample function")
        chosen_index = myth_index_sample_function(len(myth_embeddings))
        logger.debug(f"Selected myth at index {chosen_index} using sample function")
        return chosen_index

def find_most_similar_listener_myth(
    target_embedding: Embedding,
    listener_myth_embeddings: Embeddings,
    distance_function: Callable[[Embedding, Embedding], float]
) -> Tuple[int, float]:
    """Find the most similar myth in the listener's memory."""
    if not listener_myth_embeddings:
        logger.info("Listener has no myths, will create a new myth")
        # Return a special value to indicate we should create a new myth
        return -1, 1.0  # -1 indicates no existing myth, 1.0 is maximum distance
    
    distances = [
        distance_function(target_embedding, myth_embedding) 
        for myth_embedding in listener_myth_embeddings
    ]
    chosen_index = np.argmin(distances)
    return chosen_index, distances[chosen_index]
