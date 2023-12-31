# Instructions on running the experiments

## Environments
The experiments require [Conda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html) to be installed in the system.
Two virtual environments can be prepared by running the following two commands.
```bash
conda create --name=cvar_sensing python=3.10 --yes
conda create --name=asac_sensing python=3.7.10 --yes
```

## Install the package
For most of the experiments, we use the first environment.
```bash
conda activate cvar_sensing
pip install "cvar_sensing[benchmarks] @ git+https://github.com/yvchao/cvar_sensing.git"
# OR if you have the repository cloned locally, from inside the repository directory run:
pip install .[benchmarks]
```

The baseline of [ASAC](https://github.com/vanderschaarlab/mlforhealthlabpub/tree/main/alg/asac) requires TensorFlow 1.5.
We install its dependencies in a separate environment.
```bash
conda activate asac_sensing
pip install "cvar_sensing[asac] @ git+https://github.com/yvchao/cvar_sensing.git"
# OR if you have the repository cloned locally, from inside the repository directory run:
pip install .[benchmarks]
```

## Check experiment configs
Open [exp_config.py](./exp_config.py) and update the values for
* `venv`
* `venv_asac`
* `_conda_path`

if necessary.

## Run the experiment
All experiments will be conducted by running the script of "./run_experiment.py".
```bash
conda activate base
python ./run_experiment.py
```
> [!NOTE]
> ASAC relies on an older version of TensorFlow. The results from the ASAC baseline are unstable even with the same random seeds. Currently we find no treat for this issue.

## Check the results
Analysis of the experimental results can be generated and examined with files in the [reports](../reports) folder.
