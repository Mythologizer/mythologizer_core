from typing import Tuple
import numpy as np
from mythologizer_core.types import id_list, Mythmatrix, Embeddings, Embedding
from mythicalgebra import compute_myth_embedding



def mutate_myth(myth_matrix: Mythmatrix, offset: Embeddings, mytheme_ids: id_list,*, muation_probabilities: Tuple[float, float, float] = (0.2, 0.7, 0.3)) -> Tuple[Mythmatrix, id_list, Embedding]:
    """
    Mutate a myth matrix with a given offset.
    
    Args:
        myth_matrix: The original myth matrix. Array with shape (N, 2D+1): [embeddings | offsets | weights].
        offset: The offset to apply
        mytheme_ids: The mytheme IDs
        
    Returns:
        Tuple of (mutated_matrix, mutated_mytheme_ids, mutated_embedding)
    """
      # Handle case where mytheme_ids might be a string or other format
    if not (isinstance(mytheme_ids, list) or isinstance(mytheme_ids, np.ndarray)):
        raise ValueError(f"mytheme_ids must be a list or numpy array, got {type(mytheme_ids)}")

    if len(myth_matrix) != len(mytheme_ids):
        raise ValueError(f"Number of rows in myth_matrix ({len(myth_matrix)}) must be equal to number of mytheme_ids ({len(mytheme_ids)})")

    if len(myth_matrix[0]) != 2*len(offset) + 1:
        raise ValueError("Number of columns in myth_matrix must be 2*(len offset) + 1")
    
    prob_delete_mytheme, prob_mutating_mythemes, prob_reordering_mythemes = muation_probabilities
    

    if np.random.random() > prob_delete_mytheme and len(myth_matrix) > 1:
        delete_index = np.random.randint(0, len(myth_matrix))
        myth_matrix = np.delete(myth_matrix, delete_index, axis=0)
        mytheme_ids = np.delete(mytheme_ids, delete_index)

    elif np.random.random() > prob_mutating_mythemes:
        # get random mask which mythemes to mutate
        mutate_mask = np.random.rand(len(myth_matrix)) < prob_mutating_mythemes
        # mutate the mythemes by adding offset to the offset columns
        offset_start_col = len(offset)  # Start of offset columns
        myth_matrix[mutate_mask, offset_start_col:2*len(offset)] += offset * np.random.random(len(offset))
        # mutate weights (single float last column) by adding a random delta between -0.5 and 0.5 and clamp to 0.1 and 1
        weight_deltas = (np.random.random(np.sum(mutate_mask)) * 2 - 1) * 0.5
        myth_matrix[mutate_mask, -1] += weight_deltas
        # Ensure weights stay positive and have minimum value
        myth_matrix[mutate_mask, -1] = np.clip(myth_matrix[mutate_mask, -1], 0.1, 1.0)

    elif np.random.random() < prob_reordering_mythemes:
        mask = np.random.rand(len(myth_matrix)) < prob_reordering_mythemes
        # get the indices where True
        true_indices = np.where(mask)[0]
        if len(true_indices) > 1:  # Only reorder if we have at least 2 elements
            # shuffle those indices
            shuffled = np.random.permutation(true_indices)
            # apply reordering consistently
            myth_matrix[true_indices] = myth_matrix[shuffled]
            # Handle mytheme_ids reordering safely
            if isinstance(mytheme_ids, np.ndarray):
                mytheme_ids[true_indices] = mytheme_ids[shuffled]
            else:
                # Convert to list, reorder, then convert back to original type
                mytheme_ids_list = list(mytheme_ids)
                for i, j in zip(true_indices, shuffled):
                    mytheme_ids_list[i] = mytheme_ids[j]
                mytheme_ids = type(mytheme_ids)(mytheme_ids_list)

    # Ensure all weights are positive
    myth_matrix[:, -1] = np.maximum(myth_matrix[:, -1], 0.1)
    
    w_sum = myth_matrix[:, -1].sum()
    if w_sum > 0:
        myth_matrix[:, -1] /= w_sum
    else:
        # If somehow weights are still zero, set equal weights
        myth_matrix[:, -1] = 1.0 / len(myth_matrix)
    
    # Calculate the mutated myth embedding using compute_myth_embedding from mythicalgebra
    mutated_embedding = compute_myth_embedding(myth_matrix)
    
    return myth_matrix, mytheme_ids, mutated_embedding