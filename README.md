# KGQA

This repository implements some Knowledge Graph-based Question Answering (KGQA) methods, including:

- Raw LLM
- Naive RAG
- ToG (Tool over Graph)
- FastToG (Efficient ToG variant)

## Directory Structure Overview

```bash
.
├── construction/       # Knowledge graph construction (e.g., NER-based)
├── data/               # Datasets used for evaluation
├── data_loader/        # Parsing and loading input data
├── graph/              # KG classes and community algorithms
├── lm/                 # Large language model interfaces
├── qa/                 # Reasoning and answering tools (RAG, ToG, FastToG)
├── utils/              # Similarity and utility functions
├── example_run.py      # Example
├── README.md
└── requirements.txt
```
## Installation

Clone the repository:

```bash
git clone <this-repo-url>
cd <this-repo-directory>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Example Usage
For a minimal and modular example of how to integrate the core components (data loader, KG constructor, QA methods), you can refer to `example_run.py`.

```bash
python example_run.py \
    --method ToG \           # QA method: rawLLM, RAG, ToG, FastToG
    --query-ids 94,1041 \    # Comma-separated, available id range: 0 - 2416
    --output-dir ./exp_data  # Directory to store outputs and logs
```

This script demonstrates how to:

- Use MuSiQueDataLoader to load a subset of queries
- Construct a KG for each query using NERConstructor
- Choose one of several QA methods (rawLLM, RAG, ToG, FastToG, etc.)
- Log results and outputs to a specified directory

You can treat `example_run.py` as a template for building custom pipelines or adapting experiments to your own datasets.

Note: Each class (such as ToG, FastToG, LLM, etc.) supports additional optional parameters beyond those shown in the example. You are encouraged to explore their constructors to customize behaviors.