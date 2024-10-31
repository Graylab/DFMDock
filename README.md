# DFMDock
DFMDock (Denoising Force Matching Dock), a diffusion model that unifies sampling and ranking within a single framework.

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Graylab/DFMDock.git
cd DFMDock
```

### 2. Create and Activate Conda Environment

Run the following commands to create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate DFMDock
```

### 3. Install the Project in Editable Mode

To install the project in editable mode, run the following command:

```bash
pip install -e .
```


### Usage

To run inference on your own PDB files, use the following command:

```bash
python src/inference_single.py path_to_input_pdb_1 path_to_input_pdb_2
```

### Citing this work

```bibtex
@article{chu2024unified,
  title={Unified Sampling and Ranking for Protein Docking with DFMDock},
  author={Chu, Lee-Shin and Sarma, Sudeep and Gray, Jeffrey J},
  journal={bioRxiv},
  pages={2024--09},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```


