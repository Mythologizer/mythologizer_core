from mythologizer_core.types import EmbeddingFunction
from mythologizer_core import AgentAttribute
from mythologizer_postgres.db import apply_schemas, drop_everything
from mythologizer_postgres.connectors.mytheme_store import insert_mythemes_bulk
from mythologizer_postgres.connectors import insert_agent_attribute_defs

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


def setup_simulation(embedding_function: EmbeddingFunction, mythemes: list[str], agent_attributes: list[AgentAttribute]) -> None:
    
    embedding_dim = get_embedding_dim(embedding_function)
    logger.info(f"Embedding dimension: {embedding_dim}")

    drop_everything()
    logger.info("Dropped all tables")
    apply_schemas(embedding_dim)
    logger.info(f"Applied schemas with embedding dimension: {embedding_dim}")

    logger.info("Computing mytheme embeddings")
    mytheme_embeddings = [embedding_function(mytheme) for mytheme in mythemes]
    
    logger.info("Inserting mytheme embeddings into database")
    insert_mythemes_bulk(mythemes,mytheme_embeddings)

    logger.info("Inserting agent attribute definitions into database")
    insert_agent_attribute_defs(agent_attributes)

    logger.info("Finished")







