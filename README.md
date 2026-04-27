# CIRA NLP Project

This project contains the extraction of the CIRA NLP project pipeline.
The codebase is structured around several modular components that handle embeddings, short-term and long-term memory operations, and SLM orchestration.

## Features

- **Encoder**: Produces semantic embeddings (BGE by default).
- **WMB (Working Memory Buffer)**: Manages short-term context.
- **LTMS (Long-Term Memory Store)**: Manages persistent data and knowledge.
- **Orchestrator**: Ties the components together for final inference and context resolution.

## Setup & Local Execution

The project requires several machine learning libraries including PyTorch, spaCy, and Transformers. See the `requirements.txt` file for the exact dependencies.

To install dependencies:
```bash
pip install -r requirements.txt
```

You also need to install the Spacy model:
```bash
python -m spacy download en_core_web_trf
```

## Running Locally

The `config.py` file has been adjusted to provide **graceful GPU fallbacks**.
If you run this locally on a machine with 0 GPUs, the models will default to `cpu`. If you have 1 GPU, it will map everything to `cuda:0`.

## Architecture
- `src/config.py`: Device mapping, model constants, capacities.
- `src/encoder.py`: Manages the embedding model.
- `src/wmb.py`: Working Memory Buffer definitions.
- `src/ltms.py`: Long-Term Memory Store mechanisms.
- `src/interference.py`, `src/resolver.py`, `src/reconstructor.py`: Pipeline components.
- `src/orchestrator.py`: Entry point for end-to-end processing.
