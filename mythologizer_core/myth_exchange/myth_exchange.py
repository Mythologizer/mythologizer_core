from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import logging

from mythologizer_postgres.connectors import get_myth_ids_and_retention_from_agents_memory, get_agents_cultures_bulk, update_retentions_and_reorder
from mythologizer_postgres.connectors.mythicalgebra import get_myth_embeddings, get_myth_matrices_and_embedding_ids, update_myth_with_retention, insert_myth_to_agent_memory

from mythologizer_core.types import Embedding

logger = logging.getLogger(__name__)

def standard_remember_function(number_of_myths_in_memory: int,*,recollection: float = 0.7, creativity: float = 0.3) -> int:
    if number_of_myths_in_memory <= 0:
        logger.error("Length must be positive, got %d", number_of_myths_in_memory)
        raise ValueError("Length must be positive.")

    if not (0 <= recollection <= 1):
        logger.error("Recollection attribute out of bounds: %f", recollection)
        raise ValueError("Recollection attribute must be between 0 and 1.")
    if not (0 <= creativity <= 1):
        logger.error("Creativity attribute out of bounds: %f", creativity)
        raise ValueError("Creativity attribute must be between 0 and 1.")

    scale_factor = (1 - recollection) * 20 + 1
    # Use indices 1..length to compute weights, then adjust back to 0-index.
    indices = np.arange(1, number_of_myths_in_memory + 1)
    probabilities = np.exp(-scale_factor * indices / number_of_myths_in_memory)
    probabilities = (1 - creativity) * probabilities + creativity * (1 / number_of_myths_in_memory)
    probabilities /= probabilities.sum()  # Normalize

    selected = int(np.random.choice(indices, p=probabilities))
    selected_index = selected - 1  # Adjust for 0-indexing
    logger.debug(
        "standard_remember_function: Selected index %d (raw value %d) with probabilities: %s",
        selected_index, selected, probabilities,
    )
    return selected_index


def cosine_similarity(vector_a: List[float], vector_b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))


def average_embeddings(embeddings: List[Embedding]) -> Embedding:
    """Calculate the average of a list of embeddings."""
    return np.mean(embeddings, axis=0)


# Shell functions for missing implementations
def mutate_myth(myth_matrix, offset, mytheme_ids) -> Tuple:
    """
    Mutate a myth matrix with a given offset.
    
    Args:
        myth_matrix: The original myth matrix
        offset: The offset to apply
        mytheme_ids: The mytheme IDs
        
    Returns:
        Tuple of (mutated_matrix, mutated_mytheme_ids, mutated_embedding)
    """
    # TODO: Implement myth mutation logic
    pass


def combine_myths(speaker_matrix, speaker_mytheme_ids, listener_matrix, listener_mytheme_ids) -> Tuple:
    """
    Combine two myths based on their similarity.
    
    Args:
        speaker_matrix: The speaker's myth matrix
        speaker_mytheme_ids: The speaker's mytheme IDs
        listener_matrix: The listener's myth matrix
        listener_mytheme_ids: The listener's mytheme IDs
        
    Returns:
        Tuple of (combined_matrix, combined_embedding, listener_retention)
    """
    # TODO: Implement myth combination logic
    pass


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


def _get_agent_myths(agent_id: int) -> Tuple[List[int], List[float], List[Embedding]]:
    """Get myth IDs, retentions, and embeddings for an agent."""
    logger = logging.getLogger(__name__)
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


def _select_speaker_myth(
    myth_embeddings: List[Embedding],
    event_embedding: Optional[Embedding],
    myth_index_sample_function: Callable[[int], int],
    distance_function: Callable[[Embedding, Embedding], float]
) -> int:
    """Select the most appropriate myth from the speaker's memory."""
    logger = logging.getLogger(__name__)
    
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


def _get_culture_offsets(
    agent_ids: List[int],
    embedding_function: Callable[[str], Embedding],
    culture_weight: float
) -> Tuple[Embedding, Embedding]:
    """Calculate culture offsets for both agents."""
    logger = logging.getLogger(__name__)
    
    try:
        agent_cultures = get_agents_cultures_bulk(agent_ids)
        logger.debug(f"Retrieved cultures for agents {agent_ids}")
    except Exception as e:
        error_msg = f"Failed to retrieve cultures for agents {agent_ids}: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e
    
    def calculate_culture_offset(culture_ids):
        culture_embeddings = [
            embedding_function(description) 
            for _, _, description in culture_ids
        ]
        return average_embeddings(culture_embeddings) * culture_weight
    
    listener_culture_offset = calculate_culture_offset(agent_cultures[0])
    speaker_culture_offset = calculate_culture_offset(agent_cultures[1])
    
    return listener_culture_offset, speaker_culture_offset


def _calculate_agent_offsets(
    listener_agent_values: List[float],
    speaker_agent_values: List[float],
    embeddings_of_attribute_names: List[str],
    weight_of_attribute_embeddings: float,
    listener_culture_offset: Embedding,
    speaker_culture_offset: Embedding,
    event_embedding: Optional[Embedding],
    event_weight: float
) -> Tuple[Embedding, Embedding]:
    """Calculate the total offsets for both agents."""
    listener_attribute_offset = (
        np.array(listener_agent_values) * 
        np.array(embeddings_of_attribute_names) * 
        weight_of_attribute_embeddings
    )
    speaker_attribute_offset = (
        np.array(speaker_agent_values) * 
        np.array(embeddings_of_attribute_names) * 
        weight_of_attribute_embeddings
    )
    
    event_offset = event_embedding * event_weight if event_embedding is not None else 0
    
    listener_offset = listener_attribute_offset + listener_culture_offset + event_offset
    speaker_offset = speaker_attribute_offset + speaker_culture_offset + event_offset
    
    return listener_offset, speaker_offset


def _find_most_similar_listener_myth(
    target_embedding: Embedding,
    listener_myth_embeddings: List[Embedding],
    distance_function: Callable[[Embedding, Embedding], float]
) -> Tuple[int, float]:
    """Find the most similar myth in the listener's memory."""
    distances = [
        distance_function(target_embedding, myth_embedding) 
        for myth_embedding in listener_myth_embeddings
    ]
    chosen_index = np.argmin(distances)
    return chosen_index, distances[chosen_index]


def _update_listener_memory(
    listener_agent_id: int,
    combined_myth_matrix,
    combined_myth_embedding,
    chosen_listener_myth_id: int,
    chosen_listener_mytheme_ids,
    retention_listener: float,
    distance_to_listener_myth: float,
    new_myth_threshold: float
) -> None:
    """Update the listener's memory with the combined myth."""
    logger = logging.getLogger(__name__)
    
    if distance_to_listener_myth > new_myth_threshold:
        # Insert new myth when distance exceeds threshold
        logger.info(f"Distance {distance_to_listener_myth} exceeds threshold {new_myth_threshold}, inserting new myth")
        try:
            insert_myth_to_agent_memory(
                listener_agent_id, 
                combined_myth_matrix, 
                chosen_listener_mytheme_ids, 
                combined_myth_embedding
            )
            logger.info(f"Successfully inserted new myth for listener {listener_agent_id}")
        except Exception as e:
            error_msg = f"Failed to insert new myth for listener {listener_agent_id}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    else:
        # Update existing myth
        logger.info(f"Distance {distance_to_listener_myth} within threshold {new_myth_threshold}, updating existing myth")
        try:
            update_myth_with_retention(
                listener_agent_id,
                chosen_listener_myth_id,
                combined_myth_matrix,
                chosen_listener_mytheme_ids,
                retention_listener,
                combined_myth_embedding
            )
            logger.info(f"Successfully updated existing myth {chosen_listener_myth_id} for listener {listener_agent_id}")
        except Exception as e:
            error_msg = f"Failed to update myth {chosen_listener_myth_id} for listener {listener_agent_id}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e


def _update_speaker_retention(
    speaker_agent_id: int,
    myth_ids: List[int],
    retentions: List[float],
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


def tell_myth(
    listener_agent_id: int,
    speaker_agent_id: int,
    listener_agent_values: List[float],
    speaker_agent_values: List[float],
    embeddings_of_attribute_names: List[str],
    embedding_function: Callable[[str], Embedding],
    *,
    event_embedding: Optional[Embedding] = None,
    event_weight: float = 0.0,
    culture_weight: float = 0.0,
    weight_of_attribute_embeddings: float = 1.0,
    new_myth_threshold: float = 0.5,
    retention_remember_factor: float = 0.1,
    retention_forget_factor: float = 0.05,
    myth_index_sample_function: Callable[[int], int] = standard_remember_function,
    distance_function: Callable[[Embedding, Embedding], float] = cosine_similarity,
    culture_embedding_dict: Dict[int, Embedding] = None
) -> None:
    """
    Tell a myth from speaker to listener, updating both agents' memories.
    
    Args:
        listener_agent_id: ID of the agent receiving the myth
        speaker_agent_id: ID of the agent telling the myth
        listener_agent_values: List of attribute values for the listener
        speaker_agent_values: List of attribute values for the speaker
        embeddings_of_attribute_names: List of attribute name embeddings
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
    speaker_myth_ids, speaker_retentions, speaker_myth_embeddings = _get_agent_myths(speaker_agent_id)
    listener_myth_ids, listener_retentions, listener_myth_embeddings = _get_agent_myths(listener_agent_id)
    logger.info(f"Speaker has {len(speaker_myth_ids)} myths, Listener has {len(listener_myth_ids)} myths")
    logger.debug(f"Speaker myth IDs: {speaker_myth_ids}")
    logger.debug(f"Listener myth IDs: {listener_myth_ids}")
    
    # Check if speaker has any myths
    if len(speaker_myth_ids) == 0:
        logger.info("Speaker has no myths to share, skipping myth exchange")
        return
    
    # Select the most appropriate myth from speaker's memory
    logger.info("Selecting most appropriate myth from speaker's memory...")
    chosen_speaker_myth_index = _select_speaker_myth(
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
    listener_culture_offset, speaker_culture_offset = _get_culture_offsets(
        [listener_agent_id, speaker_agent_id],
        embedding_function,
        culture_weight
    )
    logger.debug(f"Culture offsets calculated - listener: {listener_culture_offset.shape}, speaker: {speaker_culture_offset.shape}")
    
    # Calculate total offsets for both agents
    logger.info("Calculating total agent offsets...")
    listener_offset, speaker_offset = _calculate_agent_offsets(
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
    
    # Mutate the speaker's myth with their offset
    logger.info("Mutating speaker's myth with agent offset...")
    mutated_speaker_myth_matrix, mutated_speaker_mytheme_ids, mutated_speaker_myth_embedding = mutate_myth(
        chosen_speaker_myth_matrix, 
        speaker_offset, 
        chosen_speaker_mytheme_ids
    )
    logger.debug(f"Speaker myth mutated successfully")
    
    # Find the most similar myth in listener's memory
    logger.info("Finding most similar myth in listener's memory...")
    chosen_listener_myth_index, distance_to_listener_myth = _find_most_similar_listener_myth(
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
        chosen_listener_mytheme_ids
    )
    logger.debug(f"Listener myth mutated successfully")
    
    # Combine the myths
    logger.info("Combining speaker and listener myths...")
    combined_myth_matrix, combined_myth_embedding, retention_listener = combine_myths(
        mutated_speaker_myth_matrix,
        mutated_speaker_mytheme_ids,
        mutated_listener_myth_matrix,
        mutated_listener_mytheme_ids
    )
    logger.info(f"Combined myth created with listener retention: {retention_listener}")
    
    # Update listener's memory
    logger.info("Updating listener's memory...")
    _update_listener_memory(
        listener_agent_id,
        combined_myth_matrix,
        combined_myth_embedding,
        chosen_listener_myth_id,
        chosen_listener_mytheme_ids,
        retention_listener,
        distance_to_listener_myth,
        new_myth_threshold
    )
    
    # Update speaker's retention
    logger.info("Updating speaker's retention values...")
    _update_speaker_retention(
        speaker_agent_id,
        speaker_myth_ids,
        speaker_retentions,
        chosen_speaker_myth_index,
        retention_remember_factor,
        retention_forget_factor
    )
    
    logger.info(f"Myth exchange completed successfully: speaker {speaker_agent_id} -> listener {listener_agent_id}")