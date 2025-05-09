
# 🕹️ Flow Game Engine (Full-State Flow Matching)

This repository contains a **toy implementation** of a **Flow Matching model** used as a **game engine**. The model operates on *fully observable* states, learning to transition a given state \( x_t \) conditioned on an action \( a \) to the next state \( x_{t+1} \). Allowing it's utilization as a game engine. 

### Extension to Partially Observable Processes

This framework can be easily extended to partially observable settings by learning the transition distribution from a history of past states \( x_t, x_{t-1}, x_{t-2}, \ldots \) to the next state \( x_{t+1} \).  
Such a formulation enables modeling in environments where the full current state is not observable, but recent state history provides sufficient information for prediction.

## 🧰 Installation

To set up the environment:

```bash
# Install dependencies
conda env create -f environment.yaml

# Activate environment
conda activate flow_game_engine

# Install project in "editable" mode
conda install conda-build  # if not already installed
conda develop .
```

## Quick start

```bash

# Generate Dataset
python scripts/get_data.py

# Train model
python scripts/train.py

# Play

python scripts/play.py
```

### Project structure

.
├── scripts/            # Scripts to run data collection, training, etc.
│   └── get_data.py
├── src/                # Main source code
│   ├── agent/
│   ├── game/
│   ├── models/
│   ├── utils/
├── data/               # Collected datasets
├── environment.yaml    # Conda environment definition
├── setup.py            # For editable install
└── README.md


