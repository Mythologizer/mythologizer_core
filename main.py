import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from sentence_transformers import SentenceTransformer
from mythologizer_core import setup_simulation
from mythologizer_core.types import Embedding

def standard_embedding_function(text: str) -> Embedding:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(text)

def main():
    setup_simulation(standard_embedding_function)

if __name__ == "__main__":
    main()