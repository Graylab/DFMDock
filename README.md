# DFMDock
DFMDock (Denoising Force Matching Dock), a diffusion model that unifies sampling and ranking within a single framework.


## Prerequisites

- [Git](https://git-scm.com/)
- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)

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

### Reproducing Results

To reproduce the results from the paper, follow these steps:

1. Navigate to the `scripts` directory:
    ```bash
    cd scripts
    ```

2. Run the inference script by executing the following command:
    ```bash
    bash inference.sh
    ```

This will initiate the process to reproduce the results as described in the paper.

### Usage

To run inference on your own PDB files, use the following command:

```bash
python src/inference_app.py path_to_input_pdb_1 path_to_input_pdb_2
```

