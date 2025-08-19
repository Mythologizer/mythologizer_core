from pydantic import BaseModel
from mythologizer_core.types import Embedding, id_type, Mythmatrix, Embeddings, Weights, Weight
from mythicalgebra import num_mythemes, decompose_myth_matrix, compute_myth_embedding, compose_myth_matrix
import numpy as np
from typing import Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mythologizer_core.queus.update_myth_queu import UpdateMythQueue


class Myth(BaseModel):
    id: id_type | None = None
    _embedding: Embedding
    _mytheme_embedding_ids: list[id_type]
    _mythmatrix: Mythmatrix
    _myth_store_insert_func: Callable
    _update_queue: 'UpdateMythQueue'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.id is None:
            self.id = self._myth_store_insert_func()
            self.id = self.id
        else:
            self.id = self.id

    def _update(self) -> None:
        self.embedding = compute_myth_embedding(self.mythmatrix)
        self._update_queue.enqueue(self)


    def num_mythemes(self) -> int:
        return num_mythemes(self.mythmatrix)
    
    def decompose_myth_matrix(self) -> Tuple[Embedding, Embedding, Weights]:
        return decompose_myth_matrix(self.mythmatrix)
    
    def update_myth_embedding(self) -> None:
        self.embedding = compute_myth_embedding(self.mythmatrix)
    
    @property
    def offsets(self) -> Embeddings:
        _, offsets, _ = self.decompose_myth_matrix()
        return offsets

    @offsets.setter
    def offsets(self, value: Embeddings) -> None:
        value = np.array(value)
        embeddings, _, weights = self.decompose_myth_matrix()
        self.mythmatrix = compose_myth_matrix(embeddings, value, weights)
        self._update()
    
    @property
    def weights(self) -> Weights:
        _, _, weights = self.decompose_myth_matrix()
        return weights
    
    @weights.setter
    def weights(self, value: Weights) -> None:
        value = np.array(value)
        embeddings, offsets, _ = self.decompose_myth_matrix()
        self.mythmatrix = compose_myth_matrix(embeddings, offsets, value)
        self._update()

    def reorder(self, new_order: list[int]) -> None:
        self.mytheme_embedding_ids = [self.mytheme_embedding_ids[i] for i in new_order]
        self.mythmatrix = self.mythmatrix[new_order]
        self._update()
    
    def add_mytheme(self, mytheme_id: id_type, embedding: Embedding, *, offset: Embedding | None = None, weight: Weight | None = None, position: int | None = None) -> None:
        if offset is None:
            offset = np.zeros(len(embedding))
        else:
            if len(offset) != len(embedding):
                raise ValueError("offset must be the same length as embedding")
            offset = np.array(offset)
        if weight is None:
            weight = 1.0
        
        if position is None:
            self.mytheme_embedding_ids.append(mytheme_id)
            old_embeddings, old_offsets, old_weights = self.decompose_myth_matrix()
            new_embeddings = np.concatenate((old_embeddings, np.array([embedding])))
            new_offsets = np.concatenate((old_offsets, np.array([offset])))
            new_weights = np.concatenate((old_weights, np.array([weight])))
            self.mythmatrix = compose_myth_matrix(new_embeddings, new_offsets, new_weights)
            self._update()
        else:
            old_embeddings, old_offsets, old_weights = self.decompose_myth_matrix()
            new_embeddings = np.concatenate((old_embeddings[:position], np.array([embedding]), old_embeddings[position:]))
            new_offsets = np.concatenate((old_offsets[:position], np.array([offset]), old_offsets[position:]))
            new_weights = np.concatenate((old_weights[:position], np.array([weight]), old_weights[position:]))
            self.mythmatrix = compose_myth_matrix(new_embeddings, new_offsets, new_weights)
            self._update()

    def remove_mytheme(self, mytheme_id: id_type) -> None:
        index = self.mytheme_embedding_ids.index(mytheme_id)
        self.mytheme_embedding_ids.pop(index)
        # in mythmatrix remove row
        self.mythmatrix = np.delete(self.mythmatrix, index, axis=0)
        self._update()
        

    # read only properties 
    @property
    def embedding(self) -> Embedding:
        return self._embedding
    
    @property
    def mytheme_embedding_ids(self) -> list[id_type]:
        return self._mytheme_embedding_ids
    
    @property
    def mytheme_embeddings(self) -> Embeddings:
        embeddings, _, _ = self.decompose_myth_matrix()
        return embeddings
    
    @property
    def mythmatrix(self) -> Mythmatrix:
        return self._mythmatrix
