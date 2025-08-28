from typing import List, Tuple
import numpy as np
import logging

from mythologizer_core.types import Embedding, Embeddings

logger = logging.getLogger(__name__)

def combine_indices(listener: list[int], speaker: list[int]) -> list[tuple[str, int]]:
    out, nxt = [], {}

    def find(x):
        return x if x not in nxt else (nxt.__setitem__(x, find(nxt[x])) or nxt[x])

    def place(i, role):
        x = find(i)                  # shifted for ordering
        out.append((role, i, x))     # keep original i, store x for sort
        nxt[x] = find(x + 1)

    for i in listener: place(i, "listener")
    for i in speaker:  place(i, "speaker")

    # sort by shifted index, return (role, original)
    return [(r, i) for r, i, _ in sorted(out, key=lambda t: t[2])]


def get_combination_weights(distance: float, listener_ratio_at_threshold: float) -> Tuple[float, float]:
    """
    Smoothly compute blend weights for normalized distance [0,1].
 
    distance: between 0 (close) and 1 (far)
    listener_ratio_at_threshold: desired weight_listener at distance=1
 
    Returns (weight_listener, weight_speaker) with sum = 1.
    """
    d = float(np.clip(distance, 0.0, 1.0))
    listener_ratio_at_threshold = float(np.clip(listener_ratio_at_threshold, 0.0, 1.0))
 
    # Smoothstep easing (cubic)
    s = 3*d**2 - 2*d**3  
 
    weight_listener = 0.5 + (listener_ratio_at_threshold - 0.5) * s
    weight_speaker = 1.0 - weight_listener
    return weight_listener, weight_speaker

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

def cosine_similarity(vector_a: Embedding, vector_b: Embedding) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))


def average_embeddings(embeddings: Embeddings) -> Embedding:
    """Calculate the average of a list of embeddings."""
    return np.mean(embeddings, axis=0)