"""
Contains project-level constants used to configure paths and wandb logging.

Paths are configured using the `.env` file in the project root.
"""

import logging
import os
import pathlib
from loguru import logger

## path constants
# project source code
SRC_PATH = pathlib.Path(__file__).parent
# project root
PROJECT_PATH = SRC_PATH.parent

## environment variables
if not os.path.exists(PROJECT_PATH / ".env"):
    logger.debug(
        "No `.env` file found in project root. Checking for env vars..."
    )
    # If no `.env` file found, check for an env var
    if os.environ.get("DATA_PATH") is not None:
        logger.debug("Found env var `DATA_PATH`:.")
        DATA_PATH = os.environ.get("DATA_PATH")
    else:
        logger.debug("No env var `DATA_PATH` found. Setting default...")
        DATA_PATH = str(SRC_PATH / "data")
        os.environ["DATA_PATH"] = str(DATA_PATH)
else:
    import dotenv  # lazy import to avoid dependency on dotenv

    dotenv.load_dotenv(PROJECT_PATH / ".env")

    DATA_PATH = os.environ.get("DATA_PATH")
    """Root path to the data directory. """

logger.info(f"DATA_PATH: {DATA_PATH}")
# set default values in case .env or hydra not used
if os.environ.get("ROOT_DIR") is None:
    ROOT_DIR = str(PROJECT_PATH)
    os.environ["ROOT_DIR"] = str(ROOT_DIR)
if os.environ.get("DATA_PATH") is None:
    DATA_PATH = str(PROJECT_PATH / "data")
    os.environ["DATA_PATH"] = str(DATA_PATH)
if os.environ.get("RUNS_PATH") is None:
    RUNS_PATH = str(PROJECT_PATH / "runs")
    os.environ["RUNS_PATH"] = str(RUNS_PATH)

# hydra constants
HYDRA_CONFIG_PATH = SRC_PATH / "config"
WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT")

# logging constants
DEFAULT_LOG_FORMATTER = logging.Formatter(
    "%(asctime)s %(levelname)s: %(message)s [in %(funcName)s at %(pathname)s:%(lineno)d]"
)
DEFAULT_LOG_FILE = PROJECT_PATH / "logs" / "default_log.log"
DEFAULT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
DEFAULT_LOG_LEVEL = logging.INFO  # no verbose logging as default
