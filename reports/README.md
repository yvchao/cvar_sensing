# Analysis of the experiment results

A concise summary of the experimental results is provided in "analysis.md".
The content in that file can be reproduced by following the instructions below.
```bash
# move to the "experiments" folder
cd ../experiments

# run the experiments
conda activate cvar_sensing
python run_experiment.py

# install the quarto-cli tool
conda install -c conda-forge quarto

# reproduce the experiment summary
quarto render .
``
