## Global Anchors

# Setup
Setup environment variables by copying the `.env.example` file, renaming it to `.env`, and filling in the values.

This repository uses `spacy`. Make sure to run `python -m spacy download en_core_web_sm`.

This repository also uses `wandb` as a logger, make sure to put your wandb information into the `.env`.

# Generating Explanations
See `globalanchors/explain.py` for how explanations are generated.

# Tests
Tests are run using `pytest`.

Just put in the command line: `python -m pytest tests/` from the project directory.