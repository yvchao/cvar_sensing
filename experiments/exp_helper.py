import os
import subprocess
import sys
from pathlib import Path

from .exp_config import conda_path


def get_conda_env():
    # Get current python executable's conda environment name.
    python_executable_path = sys.executable
    env_path = os.path.dirname(os.path.dirname(python_executable_path))
    if f"{conda_path}" in env_path:
        env_name = os.path.basename(env_path)
        return env_name
    else:
        return "Not in a Conda environment"


def get_cmd(
    script: Path | str,
    venv: str,
    gpu: int | None = 0,
    epochs: int | None = 200,
    parameters: dict[str, int | float] = {},
):
    script = Path(script)
    env = get_conda_env()
    if env != venv:
        cmd = f"{conda_path} run -n {venv} python".split(" ")
    else:
        cmd = ["python"]
    cmd += [script.as_posix()]
    if gpu is not None:
        cmd += ["--gpu", f"{gpu}"]
    if epochs is not None:
        cmd += ["--epochs", f"{epochs}"]
    for k, v in parameters.items():
        cmd += [f"--{k}", f"{v}"]

    return cmd


def run_script(cmd: list[str], exp_subdir: str):
    # Get the experiment python script file stem name.
    script_base = Path([x for x in cmd if x.endswith(".py")][0]).stem
    # Get any non gpu/epoch parameters & their values to append.
    params = "_".join(
        [
            f"{x.replace('--', '')}={cmd[idx + 1]}"
            for idx, x in enumerate(cmd)
            if x.startswith("--")
            if x not in ("--gpu", "--epochs")
        ]
    )
    # Get working directory - assumed to be the parent directory of this script.
    wd = Path(__file__).parent / exp_subdir
    # Construct the log file path using all above details.
    log_file = wd / Path(f"{script_base + ('_' + params if params != '' else '')}.log")
    log_file.touch(exist_ok=True)
    # Print user info and run the script.
    print(f"    Running command: {' '.join(cmd)}")
    print(f"    In working directory: {wd}")
    print(f"    Output logged to: {log_file}")
    with open(log_file, "w") as f:
        return subprocess.Popen(cmd, stdout=f, stderr=f)
