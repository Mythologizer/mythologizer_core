from mythologizer_core.types import EmbeddingFunction
from mythologizer_core import AgentAttribute
from typing import List
import logging

logger = logging.getLogger(__name__)

def test_setup_simulation(
    embedding_function: EmbeddingFunction | str,
    mythemes: List[str] | str,
    agent_attributes: List[AgentAttribute]
) -> None:
    """
    Test function with same parameters as setup_simulation but just logs hello world.
    
    Args:
        embedding_function: Function to generate embeddings from text (ignored)
        mythemes: List of mytheme strings to embed (ignored)
        agent_attributes: List of agent attribute definitions (ignored)
    """
    print("Hello World from test_setup_simulation!")
    logger.info("Hello World from test_setup_simulation!")
    
    print(f"Received parameters:")
    print(f"  embedding_function: {embedding_function}")
    print(f"  mythemes: {mythemes}")
    print(f"  agent_attributes count: {len(agent_attributes)}")
    
    logger.info(f"Received parameters: embedding_function={embedding_function}, mythemes={mythemes}, agent_attributes_count={len(agent_attributes)}")
    
    # Simulate some work
    import time
    time.sleep(2)
    
    print("Test setup simulation completed!")
    logger.info("Test setup simulation completed!")

