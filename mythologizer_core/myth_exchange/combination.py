from typing import Tuple
import numpy as np
from mythologizer_core.types import id_list, Mythmatrix, Embedding
from mythicalgebra import compute_myth_embedding
from mythologizer_core.myth_exchange.utils import get_combination_weights, combine_indices

def combine_myths(
    speaker_matrix: Mythmatrix,
    speaker_mytheme_ids: id_list,
    listener_matrix: Mythmatrix,
    listener_mytheme_ids: id_list,
    distance_to_listener_myth: float,
    *,
    max_weight_for_combination_listener: float = 0.7,
) -> Tuple[Mythmatrix, id_list, Embedding]:
    """
    Combine two myths by sampling rows from speaker and listener, preserving per-role order,
    then normalizing the weights column.

    Args:
        speaker_matrix: shape (N, 2D+1) [embeddings | offsets | weights]
        speaker_mytheme_ids: shape (N,)
        listener_matrix: shape (M, 2D+1) [embeddings | offsets | weights]
        listener_mytheme_ids: shape (M,)
            distance_to_listener_myth: distance in [0, max_weight_for_combination_listener]
    max_weight_for_combination_listener: threshold parameter for weighting

    Returns:
        combined_matrix: stacked rows in merged order
        combined_mytheme_ids: corresponding ids in the same order
        # If you need retention: use len(indices_listener) / len(listener_matrix)
    """
    # weights must sum to 1
    weight_listener, weight_speaker = get_combination_weights(
        distance_to_listener_myth, max_weight_for_combination_listener
    )

    n_s = len(speaker_matrix)
    n_l = len(listener_matrix)

    # sample sizes with rounding and clamping
    k_s = int(round(n_s * weight_speaker))
    k_l = int(round(n_l * weight_listener))
    k_s = max(0, min(k_s, n_s))
    k_l = max(0, min(k_l, n_l))

    # ensure we pick at least one row overall
    if k_s + k_l == 0:
        if weight_listener >= weight_speaker and n_l > 0:
            k_l = 1
        elif n_s > 0:
            k_s = 1

    # sample without replacement and keep original order per role
    indices_speaker = np.sort(np.random.choice(n_s, size=k_s, replace=False)) if k_s > 0 else np.array([], dtype=int)
    indices_listener = np.sort(np.random.choice(n_l, size=k_l, replace=False)) if k_l > 0 else np.array([], dtype=int)

    # merge the two ordered index lists; must preserve per-role relative order
    combined_indices = combine_indices(indices_speaker, indices_listener)  # -> iterable of (role, old_index)

    rows = []
    ids = []
    for role, old_index in combined_indices:
        try:
            if role == "listener":
                if old_index < len(listener_matrix):
                    rows.append(listener_matrix[old_index])
                    ids.append(listener_mytheme_ids[old_index])
                else:
                    logger.warning(f"Listener index {old_index} out of bounds for matrix of size {len(listener_matrix)}")
            else:  # speaker
                if old_index < len(speaker_matrix):
                    rows.append(speaker_matrix[old_index])
                    ids.append(speaker_mytheme_ids[old_index])
                else:
                    logger.warning(f"Speaker index {old_index} out of bounds for matrix of size {len(speaker_matrix)}")
        except Exception as e:
            logger.error(f"Error accessing {role} matrix at index {old_index}: {e}")
            logger.error(f"Matrix sizes - listener: {len(listener_matrix)}, speaker: {len(speaker_matrix)}")
            logger.error(f"Indices - listener: {indices_listener}, speaker: {indices_speaker}")
            raise

    if len(rows) == 0:
        raise ValueError("No rows to combine")

    combined_matrix = np.vstack(rows)
    combined_mytheme_ids = np.asarray(ids)

    # normalize the weights column safely
    w_sum = combined_matrix[:, -1].sum()
    if w_sum > 0:
        combined_matrix[:, -1] /= w_sum
    else:
        raise ValueError("Weights error")

    # Optional if you want retention as a third return value:
    # listener_retention = len(indices_listener) / n_l if n_l > 0 else 0.0
    # return combined_matrix, combined_mytheme_ids, listener_retention

    combined_myth_embedding = compute_myth_embedding(combined_matrix)

    return combined_matrix, combined_mytheme_ids, combined_myth_embedding