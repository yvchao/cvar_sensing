# Set this to the name of the main environment (see README.md for details):
venv = "cvar_sensing"

# Set this to the name of the ASAC environment (see README.md for details):
venv_asac = "asac_sensing"

# Set this to `None` to use default `conda` command location,
# or specify the path to the `conda` command:
_conda_path = None

# Default epochs:
epochs = 200


# Do not change:
if _conda_path is None:
    conda_path = "conda"
else:
    conda_path = _conda_path
