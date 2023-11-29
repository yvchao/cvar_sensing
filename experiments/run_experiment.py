import subprocess
import sys
from pathlib import Path

# Add repo root to PYTHONPATH to ensuer we can access experiments module:
sys.path.append((Path(__file__).parent.parent).as_posix())

from experiments.exp_config import conda_path, venv


def get_free_gpu_indices():
    def run_cmd(cmd):
        out = (subprocess.check_output(cmd, shell=True)).decode("utf-8")[:-1]
        return out

    out = run_cmd("nvidia-smi -q -d Memory | grep -A4 GPU")
    out = (out.split("\n"))[1:]
    out = [l for l in out if "--" not in l]  # noqa: E741

    total_gpu_num = int(len(out) / 5)
    gpu_bus_ids = []
    for i in range(total_gpu_num):
        gpu_bus_ids.append([l.strip().split()[1] for l in out[i * 5 : i * 5 + 1]][0])  # noqa: E741

    out = run_cmd("nvidia-smi --query-compute-apps=gpu_bus_id --format=csv")
    gpu_bus_ids_in_use = (out.split("\n"))[1:]
    gpu_ids_in_use = []

    for bus_id in gpu_bus_ids_in_use:
        gpu_ids_in_use.append(gpu_bus_ids.index(bus_id))

    return [i for i in range(total_gpu_num) if i not in gpu_ids_in_use]


def run_script(script: Path | str, cwd: Path | str, gpu: int = 0):
    script = Path(script)
    cmd = f"{conda_path} run -n {venv} python".split(" ") + [
        script.as_posix(),
        "--gpu",
        f"{gpu}",
    ]
    log_file = cwd / Path(f"{script.stem}.log")
    log_file.touch(exist_ok=True)
    print(f"Running command: {' '.join(cmd)}")
    print(f"In working directory: {cwd}")
    print(f"Output logged to: {log_file}")
    with open(log_file, "w") as f:
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=f, stderr=f)
    return proc


available_gpus = get_free_gpu_indices()
folders = [
    "./exp-synthetic",
    "./exp-adni",
]
script = "run_experiment.py"

if len(available_gpus) >= 2:
    print("run experiments in parallel.")  # noqa
    procs = [run_script(script, folder, available_gpus[i]) for i, folder in enumerate(folders)]
    exit_codes = [proc.wait() for proc in procs]
else:
    print("run experiments sequentially.")  # noqa
    for folder in folders:
        proc = run_script(script, folder)
        proc.wait()
