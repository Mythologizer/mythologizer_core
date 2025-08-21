from mythologizer_core.simulation import run_simulation

from dotenv import load_dotenv, find_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv(find_dotenv())

run_simulation()
