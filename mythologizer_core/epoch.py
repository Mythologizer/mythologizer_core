from mythologizer_postgres.connectors import get_agent_attribute_matrix, update_agent_attribute_matrix, get_all_cultures
from mythologizer_core.agent_attribute import AgentAttribute
from mythologizer_core.types import EmbeddingFunction
from mythologizer_core.myth_exchange.myth_exchange import tell_myth


import logging
import numpy as np
from typing import List, Union, Optional, Tuple, Dict
import time


logger = logging.getLogger(__name__)


class Interaction:
    """Represents an interaction between a speaker and listeners."""
    
    def __init__(self, speaker: int, listeners: List[int]):
        self.speaker = speaker
        self.listeners = listeners
    
    def __str__(self) -> str:
        return f"Interaction(speaker={self.speaker}, listeners={self.listeners})"


def _get_agent_attribute_matrix_safe() -> Tuple[np.ndarray, List[int], Dict[str, int]]:
    """
    Safely retrieve the agent attribute matrix with error handling.
    
    Returns:
        Tuple of (attribute_matrix, agent_indices, attribute_name_to_col)
        
    Raises:
        RuntimeError: If database operation fails
    """
    logger.debug("Retrieving agent attribute matrix from database...")
    
    try:
        attribute_matrix, agent_indices, attribute_name_to_col = get_agent_attribute_matrix()
        logger.info(f"Successfully retrieved agent attribute matrix with {len(agent_indices)} agents and {len(attribute_name_to_col)} attributes")
        logger.debug(f"Matrix shape: {attribute_matrix.shape}")
        logger.debug(f"Agent indices: {agent_indices}")
        logger.debug(f"Attribute names: {list(attribute_name_to_col.keys())}")
        return attribute_matrix, agent_indices, attribute_name_to_col
    except Exception as e:
        error_msg = f"Failed to retrieve agent attribute matrix: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def _update_agent_attribute_matrix_safe(attribute_matrix: np.ndarray, agent_indices: List[int]) -> None:
    """
    Safely update the agent attribute matrix with error handling.
    
    Args:
        attribute_matrix: The updated attribute matrix
        agent_indices: List of agent indices
        
    Raises:
        RuntimeError: If database operation fails
    """
    logger.debug("Updating agent attribute matrix in database...")
    
    try:
        update_agent_attribute_matrix(attribute_matrix, agent_indices)
        logger.info(f"Successfully updated agent attribute matrix for {len(agent_indices)} agents")
    except Exception as e:
        error_msg = f"Failed to update agent attribute matrix: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def _get_all_cultures_safe() -> List[Tuple]:
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


def epoch_agent_attributes(agent_attributes: List[AgentAttribute]) -> Tuple[np.ndarray, List[int]]:
    """
    Apply epoch functions to agent attributes and update the database.
    
    Args:
        agent_attributes: List of agent attributes to process
        
    Returns:
        Tuple of (updated_attribute_matrix, agent_indices)
        
    Raises:
        RuntimeError: If database operations fail
    """
    logger.info("Starting agent attribute epoch processing")
    logger.debug(f"Processing {len(agent_attributes)} agent attributes")
    
    # Get current agent attribute matrix
    attribute_matrix, agent_indices, attribute_name_to_col = _get_agent_attribute_matrix_safe()
    
    # Apply epoch functions to attributes that have them
    attributes_with_epoch_functions = [attr for attr in agent_attributes if attr.epoch_change_function]
    logger.info(f"Found {len(attributes_with_epoch_functions)} attributes with epoch functions")
    
    for index, agent_attribute in enumerate(agent_attributes):
        if agent_attribute.epoch_change_function:
            logger.debug(f"Processing attribute: {agent_attribute.name}")
            
            try:
                col_index = attribute_name_to_col[agent_attribute.name]
                logger.debug(f"Attribute '{agent_attribute.name}' found at column index {col_index}")
                
                # Apply the epoch function to the specific column
                old_values = attribute_matrix[:, col_index].copy()
                attribute_matrix[:, col_index] = agent_attribute.epoch_change_function(attribute_matrix[:, col_index])
                
                # Apply min/max constraints if defined
                if agent_attribute.min is not None:
                    attribute_matrix[:, col_index] = np.maximum(attribute_matrix[:, col_index], agent_attribute.min)
                    logger.debug(f"Applied min constraint {agent_attribute.min} to {agent_attribute.name}")
                
                if agent_attribute.max is not None:
                    attribute_matrix[:, col_index] = np.minimum(attribute_matrix[:, col_index], agent_attribute.max)
                    logger.debug(f"Applied max constraint {agent_attribute.max} to {agent_attribute.name}")
                
                new_values = attribute_matrix[:, col_index]
                
                logger.info(f"Epoch function applied to '{agent_attribute.name}'")
                logger.debug(f"Values changed for {agent_attribute.name}: {old_values[:5]} -> {new_values[:5]}")
                
            except KeyError as e:
                error_msg = f"Attribute '{agent_attribute.name}' not found in attribute matrix columns"
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e
            except Exception as e:
                error_msg = f"Failed to apply epoch function to attribute '{agent_attribute.name}': {str(e)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e
    
    # Update the database with the modified matrix
    _update_agent_attribute_matrix_safe(attribute_matrix, agent_indices)
    
    logger.info("Agent attribute epoch processing completed successfully")
    return attribute_matrix, agent_indices


def get_interactions(
    agent_indices: List[int],
    number_of_interactions: int,
    max_number_of_listeners: int
) -> List[Interaction]:
    """
    Generate random interactions between agents.
    
    Args:
        agent_indices: List of available agent indices
        number_of_interactions: Number of interactions to generate
        max_number_of_listeners: Maximum number of listeners per interaction
        
    Returns:
        List of Interaction objects
        
    Raises:
        ValueError: If parameters are invalid
    """
    logger.info(f"Generating {number_of_interactions} random interactions")
    logger.debug(f"Available agents: {len(agent_indices)}, Max listeners: {max_number_of_listeners}")
    
    if not agent_indices:
        error_msg = "No agents available for interactions"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if number_of_interactions <= 0:
        error_msg = f"Number of interactions must be positive, got {number_of_interactions}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if max_number_of_listeners <= 0:
        error_msg = f"Max number of listeners must be positive, got {max_number_of_listeners}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if max_number_of_listeners >= len(agent_indices):
        logger.warning(f"Max listeners ({max_number_of_listeners}) >= available agents ({len(agent_indices)}), adjusting to {len(agent_indices) - 1}")
        max_number_of_listeners = len(agent_indices) - 1
    
    interactions = []
    for i in range(number_of_interactions):
        try:
            # Get a random speaker
            speaker = np.random.choice(agent_indices)
            
            # Get random listeners (excluding the speaker)
            available_listeners = [agent for agent in agent_indices if agent != speaker]
            num_listeners = min(max_number_of_listeners, len(available_listeners))
            
            if num_listeners > 0:
                listeners = np.random.choice(available_listeners, size=num_listeners, replace=False)
                listeners = listeners.tolist()
            else:
                listeners = []
            
            interaction = Interaction(speaker, listeners)
            interactions.append(interaction)
            
            logger.debug(f"Generated interaction {i+1}: {interaction}")
            
        except Exception as e:
            error_msg = f"Failed to generate interaction {i+1}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    logger.info(f"Successfully generated {len(interactions)} interactions")
    return interactions


def _create_culture_embedding_dict(
    cultures: List[Tuple],
    embedding_function: Union[EmbeddingFunction, str]
) -> Dict[int, np.ndarray]:
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


def _process_interaction(
    interaction: Interaction,
    attribute_matrix: np.ndarray,
    agent_indices: List[int],
    embeddings_of_attribute_names: List[np.ndarray],
    embedding_function: Union[EmbeddingFunction, str],
    culture_embedding_dict: Dict[int, np.ndarray]
) -> None:
    """
    Process a single interaction by telling myths between agents.
    
    Args:
        interaction: The interaction to process
        attribute_matrix: Matrix containing agent attribute values
        agent_indices: List of agent IDs corresponding to matrix rows
        embeddings_of_attribute_names: List of attribute name embeddings
        embedding_function: Function to convert text to embeddings
        culture_embedding_dict: Dictionary of culture embeddings
        
    Raises:
        RuntimeError: If myth exchange fails
    """
    logger.debug(f"Processing interaction: {interaction}")
    
    try:
        # Create mapping from agent IDs to matrix indices
        agent_id_to_matrix_idx = {agent_id: idx for idx, agent_id in enumerate(agent_indices)}
        
        # Map agent IDs to matrix indices
        speaker_matrix_idx = agent_id_to_matrix_idx[interaction.speaker]
        listener_matrix_indices = [agent_id_to_matrix_idx[listener_id] for listener_id in interaction.listeners]
        
        # Get agent values for this interaction
        listener_agent_values = attribute_matrix[listener_matrix_indices, :]
        speaker_agent_values = attribute_matrix[speaker_matrix_idx, :]
        
        logger.debug(f"Speaker {interaction.speaker} values shape: {speaker_agent_values.shape}")
        logger.debug(f"Listeners values shape: {listener_agent_values.shape}")
        
        # Tell myth for each listener
        for listener_idx, listener_id in enumerate(interaction.listeners):
            logger.debug(f"Telling myth from speaker {interaction.speaker} to listener {listener_id}")
            
            try:
                tell_myth(
                    listener_agent_id=listener_id,
                    speaker_agent_id=interaction.speaker,
                    listener_agent_values=listener_agent_values[listener_idx],
                    speaker_agent_values=speaker_agent_values,
                    embeddings_of_attribute_names=embeddings_of_attribute_names,
                    embedding_function=embedding_function,
                    culture_embedding_dict=culture_embedding_dict
                )
                logger.debug(f"Successfully told myth to listener {listener_id}")
                
            except Exception as e:
                error_msg = f"Failed to tell myth from speaker {interaction.speaker} to listener {listener_id}: {str(e)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e
                
    except Exception as e:
        error_msg = f"Failed to process interaction {interaction}: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def run_epoch(
    agent_attributes: List[AgentAttribute],
    embedding_function: Union[EmbeddingFunction, str],
    number_of_interactions: int,
    max_number_of_listeners: int
) -> None:
    """
    Run a complete epoch with agent attribute updates and myth exchanges.
    
    Args:
        agent_attributes: List of agent attributes to process
        embedding_function: Function to convert text to embeddings
        number_of_interactions: Number of interactions to generate
        max_number_of_listeners: Maximum number of listeners per interaction
        
    Raises:
        RuntimeError: If any operation fails
    """
    logger.info("Starting epoch execution")
    logger.info(f"Parameters: {len(agent_attributes)} attributes, {number_of_interactions} interactions, max {max_number_of_listeners} listeners")
    
    try:
        # Create embeddings for attribute names
        logger.info("Creating attribute name embeddings...")
        embeddings_of_attribute_names = []
        for attribute in agent_attributes:
            try:
                # Use the encode method for SentenceTransformer
                if hasattr(embedding_function, 'encode'):
                    embedding = embedding_function.encode(attribute.name)
                else:
                    embedding = embedding_function(attribute.name)
                embeddings_of_attribute_names.append(embedding)
                logger.debug(f"Created embedding for attribute: {attribute.name}")
            except Exception as e:
                error_msg = f"Failed to create embedding for attribute '{attribute.name}': {str(e)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e
        
        logger.info(f"Created embeddings for {len(embeddings_of_attribute_names)} attributes")
        
        # Process agent attributes
        logger.info("Processing agent attributes...")
        attribute_matrix, agent_indices = epoch_agent_attributes(agent_attributes)
        
        # Get cultures and create embeddings
        logger.info("Retrieving cultures and creating embeddings...")
        cultures = _get_all_cultures_safe()
        culture_embedding_dict = _create_culture_embedding_dict(cultures, embedding_function)
        
        # Generate interactions
        logger.info("Generating interactions...")
        interactions = get_interactions(agent_indices, number_of_interactions, max_number_of_listeners)
        
        # Process each interaction
        logger.info(f"Processing {len(interactions)} interactions...")
        successful_interactions = 0
        
        for i, interaction in enumerate(interactions):
            try:
                logger.info(f"Processing interaction {i+1}/{len(interactions)}: {interaction}")
                _process_interaction(
                    interaction,
                    attribute_matrix,
                    agent_indices,
                    embeddings_of_attribute_names,
                    embedding_function,
                    culture_embedding_dict
                )
                successful_interactions += 1
                logger.info(f"Successfully processed interaction {i+1}")
                
            except Exception as e:
                logger.error(f"Failed to process interaction {i+1}: {str(e)}")
                # Continue with other interactions even if one fails
                continue
        
        logger.info(f"Epoch completed: {successful_interactions}/{len(interactions)} interactions successful")
        
    except Exception as e:
        error_msg = f"Epoch execution failed: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e

