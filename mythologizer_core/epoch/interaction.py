from mythologizer_core.types import AgentAttributeMatrix, id_list, id_type
from mythologizer_core.types.types import Embedding, Embeddings, EmbeddingFunction
from mythologizer_core.myth_exchange.myth_exchange import tell_myth

import logging
import numpy as np
from typing import Dict, List, Union

logger = logging.getLogger(__name__)


class Interaction:
    """Represents an interaction between a speaker and listeners."""
    
    def __init__(self, speaker: id_type, listeners: id_list):
        self.speaker = speaker
        self.listeners = listeners
    
    def __str__(self) -> str:
        return f"Interaction(speaker={self.speaker}, listeners={self.listeners})"

def get_interactions(
    agent_indices: id_list,
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


def process_interaction(
    interaction: Interaction,
    attribute_matrix: AgentAttributeMatrix,
    agent_indices: id_list,
    embeddings_of_attribute_names: Embeddings,
    embedding_function: EmbeddingFunction,
    culture_embedding_dict: Dict[id_type, Embedding],
    myth_exchange_config: Dict[str, float] = None
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
                # Use default values if myth_exchange_config is None
                if myth_exchange_config is None:
                    myth_exchange_config = {
                        "event_weight": 0.0,
                        "culture_weight": 0.0,
                        "weight_of_attribute_embeddings": 1.0,
                        "new_myth_threshold": 0.5,
                        "retention_remember_factor": 0.1,
                        "retention_forget_factor": 0.05,
                        "max_weight_for_combination_listener": 0.5,
                        "mutation_probability_deletion": 0.2,
                        "mutation_probability_mutation": 0.7,
                        "mutation_probability_reordering": 0.3
                    }
                
                tell_myth(
                    listener_agent_id=listener_id,
                    speaker_agent_id=interaction.speaker,
                    listener_agent_values=listener_agent_values[listener_idx],
                    speaker_agent_values=speaker_agent_values,
                    embeddings_of_attribute_names=embeddings_of_attribute_names,
                    embedding_function=embedding_function,
                    event_weight=myth_exchange_config.get("event_weight", 0.0),
                    culture_weight=myth_exchange_config.get("culture_weight", 0.0),
                    weight_of_attribute_embeddings=myth_exchange_config.get("weight_of_attribute_embeddings", 1.0),
                    new_myth_threshold=myth_exchange_config.get("new_myth_threshold", 0.5),
                    retention_remember_factor=myth_exchange_config.get("retention_remember_factor", 0.1),
                    retention_forget_factor=myth_exchange_config.get("retention_forget_factor", 0.05),
                    max_weight_for_combination_listener=myth_exchange_config.get("max_weight_for_combination_listener", 0.5),
                    mutation_probabilities=(
                        myth_exchange_config.get("mutation_probability_deletion", 0.2),
                        myth_exchange_config.get("mutation_probability_mutation", 0.7),
                        myth_exchange_config.get("mutation_probability_reordering", 0.3)
                    ),
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