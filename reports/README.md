# Analysis of the experiment results

A concise summary of the experimental results is provided in [analysis.md](./analysis.md).
The content in that file can be reproduced by following the instructions below.
```bash
# move to the "experiments" folder
cd ../experiments

# prepare the virtual environment following the instructions in README.md
cat ./README.md

# run the experiments
conda activate cvar_sensing
python run_experiment.py

# go back to this folder
cd ../reports

# reproduce the experiment summary
quarto render .
```

> [!Note]
> [Quarto](https://quarto.org/) is required to render the final analysis.
> Please follow the instructions on their website to install the quarto-cli tool.
