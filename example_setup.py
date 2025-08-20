import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from sentence_transformers import SentenceTransformer
from mythologizer_core import setup_simulation
from mythologizer_core.types import Embedding
from mythologizer_core import AgentAttribute

def standard_embedding_function(text: str) -> Embedding:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(text)

def main():
    mythemes = ["The Hero's Journey", "Another Mytheme"]

    agent_attributes = [
    AgentAttribute(
        name='Age',
        description='Age of the agent',
        d_type=int,
        min=0,
        epoch_change_function= lambda x: x + 1
    ),
    AgentAttribute(
        name='Confidence',
        description='The confidence of the agent',
        d_type=float,
        min=0.0,
        max=1.0,
        epoch_change_function= lambda x: x + random.random()
    ),
    AgentAttribute(
        name='Emotionality',
        description='The emotionality of the agent with 0 representing a very emotionless person and 1 representing a very emotional person',
        d_type=float,
        min=0.0,
        max=1.0)
    ]

    setup_simulation(standard_embedding_function, mythemes, agent_attributes)

if __name__ == "__main__":
    main()