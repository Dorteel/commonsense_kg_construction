# Commonsense Knowledge Graph Construction with Large Language Models

This repository provides a framework for constructing commonsense knowledge graphs using large language models (LLMs). The system extracts structured knowledge about real-world concepts by querying LLMs with carefully designed prompts, then aggregates and analyzes the results.

## Overview

The pipeline operates on two main inputs:
- **Concepts**: The entities to be described (e.g., "apple", "hammer").
- **Quality Dimensions**: The domains or properties relevant to each concept (e.g., colour, shape, weight).

A key step is determining which quality dimensions are relevant for each concept. This is achieved using a "context vector", which acts as a binary filter to indicate the applicability of each dimension.

The framework supports two main use cases:
1. **Irrelevant Quality Dimensions**: When a dimension does not apply to a concept (e.g., the speed of an apple).
2. **Non-defining Features**: When a value can take any form and is not a defining feature (e.g., the colour of a mug).

The initial experiments focus on 80 concepts from the MS COCO dataset, with extensions to 1000 concepts from ImageNet-1k.

## Reproducing the Experiment

### 1. Clone the Repository

```sh
git clone https://github.com/yourusername/commonsense_kg_construction.git
cd commonsense_kg_construction
```

### 2. Install Dependencies

It is recommended to use a virtual environment:

```sh
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Prepare Input Data

- Place your concepts and property definitions in the `inputs/` directory.
- Example files: `concepts_mscoco.json`, `exp_properties.yaml`.

### 4. Run the Knowledge Extraction Pipeline

You can run the main pipeline using:

```sh
python kg_constructors/main.py
```

Or, to run batch experiments (e.g., for MS COCO):

```sh
python experiments/exp_mscoco.py
```

### 5. Analyze Results

Processed outputs and analysis scripts are located in the `analysis/` directory. For example, to preprocess and analyze results:

```sh
python analysis/preprocessing.py
python analysis/main_analysis.py
```

### 6. Visualize Results

Visualization scripts are available in `analysis/visualisations/`.

---

**Note:**  
- You may need API keys for LLM providers (e.g., OpenAI, Groq, Nebula). Set these in a `.env` file in the project root.
- Output and intermediate files are stored in the `output/`, `data/`, and `logs/` directories.

## Citation

If you use this codebase, please cite or acknowledge the repository.

---

Feel free to adjust the instructions or add more details as needed!