# Agent-based Test Support Systems (AbTSS)

[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)

![DCD overview diagram](/doc/ACCEL.svg)

This repository contains the official implementation of the paper:

**"TOWARDS AGENT-BASED TEST SUPPORT SYSTEMS: AN UNSUPERVISED ENVIRONMENT DESIGN APPROACH"**  
by *Collins O. Ogbodo, Timothy J. Rogers, Mattia Dal Borgo, and David J. Wagg*  
Preprint available on [arXiv]()

---

## Key Features

- **UPOMDP-based Formulation** — Underspecified partially observable Markov descision process.
- **Dual Curriculum Design** — Unspecified enviornment design.
- **Parameterised Test Environment** Parameterise test environment by frequency range, geometry, boundary condition, simulated damages e.t.c
- **Adaptive Sensor Placement Strategy** Learn sensor placement across enviornment parameter distribution.
- **Information-Theoretic Reward** — Maximises determinant of the FIM for informative sensor placement.
- **Spatial Correlation-Aware** — Rewards spatially well-distributed sensor configurations.
- **Case Studies** — Applied to a cantilever plate test environment.
---

## Enviornment frequency segmentation
![Damage localisation](/doc/Training_Environment_Architecture.svg)

## Setup
To install the necessary dependencies, run the following commands:
```
conda create --name dcd python=3.8
conda activate dcd
pip install -r requirements.txt
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .
cd ..
pip install pyglet==1.5.11
```
---

## Run 
```
python train.py
python eval.py
```

---
## Citation
```
@article{ogbodo2025adaptive,
  title={Adaptive Sensor Steering Strategy Using Deep Reinforcement Learning for Dynamic Data Acquisition in Digital Twins},
  author={Ogbodo, Collins O and Rogers, Timothy J and Borgo, Mattia Dal and Wagg, David J},
  journal={arXiv preprint arXiv:2504.10248},
  year={2025}
}
```
