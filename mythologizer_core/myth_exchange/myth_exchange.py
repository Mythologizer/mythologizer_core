from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import logging

from mythologizer_postgres.connectors import get_myth_ids_and_retention_from_agents_memory, get_agents_cultures_ids_bulk, update_retentions_and_reorder
from mythologizer_postgres.connectors.mythicalgebra import get_myth_embeddings, get_myth_matrices_and_embedding_ids, update_myth_with_retention, insert_myth_to_agent_memory
from mythicalgebra import compute_myth_embedding

from mythologizer_core.types import Embedding

logger = logging.getLogger(__name__)

from mythologizer_core.myth_exchange.utils import combine_indices, get_combination_weights, standard_remember_function, cosine_similarity, average_embeddings
from mythologizer_core.myth_exchange.mutation import mutate_myth
from mythologizer_core.myth_exchange.myth_connectors import get_agent_myths, insert_new_myth, update_myth_in_listener_memory, update_speaker_retention
from mythologizer_core.myth_exchange.offsets import get_culture_offsets, calculate_agent_offsets
from mythologizer_core.myth_exchange.selection import select_speaker_myth, find_most_similar_listener_myth


# Helper functions for the main tell_myth function
def _validate_input_parameters(
    embeddings_of_attribute_names: List[str],
    listener_agent_values: List[float],
    speaker_agent_values: List[float]
) -> None:
    """Validate that input parameters have consistent lengths."""
    logger = logging.getLogger(__name__)
    logger.debug(f"Validating parameter lengths - attributes: {len(embeddings_of_attribute_names)}, listener: {len(listener_agent_values)}, speaker: {len(speaker_agent_values)}")
    
    if (len(embeddings_of_attribute_names) != len(listener_agent_values) or 
        len(embeddings_of_attribute_names) != len(speaker_agent_values)):
        error_msg = (
            "The number of embeddings of attribute names must be the same as "
            "the number of values for the listener and speaker agents"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.debug("Parameter validation passed")


def tell_myth(
    listener_agent_id: int,
    speaker_agent_id: int,
    listener_agent_values: List[float],
    speaker_agent_values: List[float],
    embeddings_of_attribute_names: List[Embedding],
    embedding_function: Callable[[str], Embedding],
    *,
    event_embedding: Optional[Embedding] = None,
    event_weight: float = 0.0,
    culture_weight: float = 0.0,
    weight_of_attribute_embeddings: float = 1.0,
    new_myth_threshold: float = 0.5,
    retention_remember_factor: float = 0.1,
    retention_forget_factor: float = 0.05,
    max_threshold_for_listener_myth: float = 0.5,
    mutation_probabilities: Tuple[float, float, float] = (0.2, 0.7, 0.3),
    myth_index_sample_function: Callable[[int], int] = standard_remember_function,
    distance_function: Callable[[Embedding, Embedding], float] = cosine_similarity,
    culture_embedding_dict: Dict[int, Embedding] = None
) -> None:
    logger = logging.getLogger(__name__)
    logger.debug(f"tell_myth called with embeddings_of_attribute_names type: {type(embeddings_of_attribute_names)}")
    logger.debug(f"embeddings_of_attribute_names length: {len(embeddings_of_attribute_names) if embeddings_of_attribute_names else 'None'}")
    if embeddings_of_attribute_names:
        logger.debug(f"First embedding type: {type(embeddings_of_attribute_names[0])}")
        logger.debug(f"First embedding shape: {embeddings_of_attribute_names[0].shape if hasattr(embeddings_of_attribute_names[0], 'shape') else 'No shape'}")
    logger.info(f"Starting myth exchange: speaker {speaker_agent_id} -> listener {listener_agent_id}")
    """
    Tell a myth from speaker to listener, updating both agents' memories.
    
    Args:
        listener_agent_id: ID of the agent receiving the myth
        speaker_agent_id: ID of the agent telling the myth
        listener_agent_values: List of attribute values for the listener
        speaker_agent_values: List of attribute values for the speaker
        embeddings_of_attribute_names: List of attribute embeddings (not names)
        embedding_function: Function to convert text to embeddings
        event_embedding: Optional embedding of the current event
        event_weight: Weight for event influence on myth selection
        culture_weight: Weight for cultural influence
        weight_of_attribute_embeddings: Weight for attribute embeddings
        new_myth_threshold: Threshold for creating new myths vs updating existing
        retention_remember_factor: Factor to increase speaker's myth retention
        retention_forget_factor: Factor to decrease retention of other speaker myths
        myth_index_sample_function: Function to select myth index when no event
        distance_function: Function to calculate similarity between embeddings
        culture_embedding_dict: Dictionary of culture embeddings (unused in current implementation)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting myth exchange: speaker {speaker_agent_id} -> listener {listener_agent_id}")
    logger.debug(f"Event embedding provided: {event_embedding is not None}")
    logger.debug(f"Event weight: {event_weight}, Culture weight: {culture_weight}")
    logger.debug(f"Attribute embeddings weight: {weight_of_attribute_embeddings}")
    logger.debug(f"New myth threshold: {new_myth_threshold}")
    logger.debug(f"Retention factors - remember: {retention_remember_factor}, forget: {retention_forget_factor}")
    
    # Initialize culture_embedding_dict if None
    if culture_embedding_dict is None:
        culture_embedding_dict = {}
        logger.debug("Initialized empty culture_embedding_dict")
    
    # Validate input parameters
    logger.debug("Validating input parameters...")
    _validate_input_parameters(
        embeddings_of_attribute_names, 
        listener_agent_values, 
        speaker_agent_values
    )
    logger.debug("Input parameters validated successfully")
    
    # Get myths for both agents
    logger.info("Retrieving myths from both agents' memories...")
    speaker_myth_ids, speaker_retentions, speaker_myth_embeddings = get_agent_myths(speaker_agent_id)
    listener_myth_ids, listener_retentions, listener_myth_embeddings = get_agent_myths(listener_agent_id)
    logger.info(f"Speaker has {len(speaker_myth_ids)} myths, Listener has {len(listener_myth_ids)} myths")
    logger.debug(f"Speaker myth IDs: {speaker_myth_ids}")
    logger.debug(f"Listener myth IDs: {listener_myth_ids}")
    
    # Check if speaker has any myths
    if len(speaker_myth_ids) == 0:
        logger.info("Speaker has no myths to share, skipping myth exchange")
        return
    
    # Select the most appropriate myth from speaker's memory
    logger.info("Selecting most appropriate myth from speaker's memory...")
    chosen_speaker_myth_index = select_speaker_myth(
        speaker_myth_embeddings,
        event_embedding,
        myth_index_sample_function,
        distance_function
    )
    logger.info(f"Selected speaker myth at index {chosen_speaker_myth_index}")
    
    # Get the chosen speaker myth details
    chosen_speaker_myth_id = speaker_myth_ids[chosen_speaker_myth_index]
    chosen_speaker_myth_retention = speaker_retentions[chosen_speaker_myth_index]
    chosen_speaker_myth_embedding = speaker_myth_embeddings[chosen_speaker_myth_index]
    
    try:
        chosen_speaker_myth_matrix, chosen_speaker_mytheme_ids = get_myth_matrices_and_embedding_ids(chosen_speaker_myth_id)
        logger.info(f"Selected speaker myth ID: {chosen_speaker_myth_id} with retention: {chosen_speaker_myth_retention}")
        logger.debug(f"Speaker myth has {len(chosen_speaker_mytheme_ids)} mythemes")
    except Exception as e:
        error_msg = f"Failed to retrieve myth matrices for speaker myth {chosen_speaker_myth_id}: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e
    
    # Calculate culture offsets
    logger.info("Calculating cultural influence offsets...")
    culture_offsets = get_culture_offsets(
        [listener_agent_id, speaker_agent_id],
        culture_embedding_dict,
        culture_weight
    )
    listener_culture_offset = culture_offsets[0]
    speaker_culture_offset = culture_offsets[1]
    logger.debug(f"Culture offsets calculated - listener: {listener_culture_offset.shape}, speaker: {speaker_culture_offset.shape}")
    
    # Calculate total offsets for both agents
    logger.info("Calculating total agent offsets...")
    logger.debug(f"About to call calculate_agent_offsets with embeddings_of_attribute_names type: {type(embeddings_of_attribute_names)}")
    logger.debug(f"embeddings_of_attribute_names length: {len(embeddings_of_attribute_names) if embeddings_of_attribute_names else 'None'}")
    if embeddings_of_attribute_names:
        logger.debug(f"First embedding type: {type(embeddings_of_attribute_names[0])}")
        logger.debug(f"First embedding shape: {embeddings_of_attribute_names[0].shape if hasattr(embeddings_of_attribute_names[0], 'shape') else 'No shape'}")
    try:
        listener_offset, speaker_offset = calculate_agent_offsets(
            listener_agent_values,
            speaker_agent_values,
            embeddings_of_attribute_names,
            weight_of_attribute_embeddings,
            listener_culture_offset,
            speaker_culture_offset,
            event_embedding,
            event_weight
        )
        logger.debug(f"Total offsets calculated - listener: {listener_offset.shape}, speaker: {speaker_offset.shape}")
    except Exception as e:
        logger.error(f"Error in calculate_agent_offsets: {str(e)}")
        raise
    
    # Mutate the speaker's myth with their offset
    logger.info("Mutating speaker's myth with agent offset...")
    try:
        logger.debug(f"About to call mutate_myth with chosen_speaker_mytheme_ids: {type(chosen_speaker_mytheme_ids)}, value: {chosen_speaker_mytheme_ids}")
        mutated_speaker_myth_matrix, mutated_speaker_mytheme_ids, mutated_speaker_myth_embedding = mutate_myth(
            chosen_speaker_myth_matrix, 
            speaker_offset, 
            chosen_speaker_mytheme_ids,
            muation_probabilities=mutation_probabilities
        )
    except Exception as e:
        logger.error(f"Error in mutate_myth: {str(e)}")
        raise
    logger.debug(f"Speaker myth mutated successfully")
    
    # Find the most similar myth in listener's memory
    logger.info("Finding most similar myth in listener's memory...")
    chosen_listener_myth_index, distance_to_listener_myth = find_most_similar_listener_myth(
        mutated_speaker_myth_embedding,
        listener_myth_embeddings,
        distance_function
    )
    logger.info(f"Selected listener myth at index {chosen_listener_myth_index} with distance: {distance_to_listener_myth}")
    
    # Get the chosen listener myth details
    chosen_listener_myth_id = listener_myth_ids[chosen_listener_myth_index]
    chosen_listener_myth_retention = listener_retentions[chosen_listener_myth_index]
    chosen_listener_myth_embedding = listener_myth_embeddings[chosen_listener_myth_index]
    
    try:
        chosen_listener_myth_matrix, chosen_listener_mytheme_ids = get_myth_matrices_and_embedding_ids(chosen_listener_myth_id)
        logger.info(f"Selected listener myth ID: {chosen_listener_myth_id} with retention: {chosen_listener_myth_retention}")
        logger.debug(f"Listener myth has {len(chosen_listener_mytheme_ids)} mythemes")
    except Exception as e:
        error_msg = f"Failed to retrieve myth matrices for listener myth {chosen_listener_myth_id}: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e
    
    # Mutate the listener's myth with their offset
    logger.info("Mutating listener's myth with agent offset...")
    mutated_listener_myth_matrix, mutated_listener_mytheme_ids, mutated_listener_myth_embedding = mutate_myth(
        chosen_listener_myth_matrix, 
        listener_offset, 
        chosen_listener_mytheme_ids,
        muation_probabilities=mutation_probabilities
    )
    logger.debug(f"Listener myth mutated successfully")
    

    if distance_to_listener_myth > new_myth_threshold:
        # Combine the myths
        logger.info("Combining speaker and listener myths...")
        combined_myth_matrix, combined_mytheme_ids, combined_myth_embedding = combine_myths(
            mutated_speaker_myth_matrix,
            mutated_speaker_mytheme_ids,
            mutated_listener_myth_matrix,
            mutated_listener_mytheme_ids,
            distance_to_listener_myth
        )
        retention_listener = chosen_listener_myth_retention + retention_remember_factor * 1.2 
        logger.info(f"Combined myth created with listener retention: {retention_listener}")
        insert_new_myth(
            listener_agent_id,
            combined_myth_matrix,
            combined_mytheme_ids,
            combined_myth_embedding
        )
    else:
        retention_listener = 1.0
        update_myth_in_listener_memory(
            listener_agent_id,
            chosen_listener_myth_id,
            mutated_listener_myth_matrix,
            mutated_listener_mytheme_ids,
            mutated_listener_myth_embedding,
            retention_listener
        )



    
    # Listener memory is already updated in the if/else block above
    
    # Update speaker's retention
    logger.info("Updating speaker's retention values...")
    update_speaker_retention(
        speaker_agent_id,
        speaker_myth_ids,
        speaker_retentions,
        chosen_speaker_myth_index,
        retention_remember_factor,
        retention_forget_factor
    )
    
    logger.info(f"Myth exchange completed successfully: speaker {speaker_agent_id} -> listener {listener_agent_id}")