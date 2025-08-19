from mythologizer_core.types import EmbeddingFunction
from mythologizer_core import AgentAttribute
from mythologizer_postgres.db import apply_schemas, drop_all_tables
import logging
import numpy as np

logger = logging.getLogger(__name__)

def get_embedding_dim(embedding_function: EmbeddingFunction) -> int:
        test_embedding = embedding_function("test")
        if test_embedding is type(list):
            embedding_dim = len(test_embedding)
        elif isinstance(test_embedding, np.ndarray):
            embedding_dim = test_embedding.shape[0]
        else:
            raise ValueError(f"Unsupported embedding type: {type(test_embedding)}")
        return embedding_dim

def setup_simulation(embedding_function: EmbeddingFunction) -> None:
    
    embedding_dim = get_embedding_dim(embedding_function)
    logger.info(f"Embedding dimension: {embedding_dim}")

    drop_all_tables()
    logger.info("Dropped all tables")
    apply_schemas(embedding_dim)
    logger.info(f"Applied schemas with embedding dimension: {embedding_dim}")



