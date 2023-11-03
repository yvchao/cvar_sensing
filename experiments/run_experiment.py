import subprocess
from pathlib import Path

venv = "cvar_sensing"


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


def run_script(script: Path | str, gpu: int = 0):
    script = Path(script)
    cmd = f"conda run -n {venv} python".split(" ") + [
        script.as_posix(),
        "--gpu",
        f"{gpu}",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc


available_gpus = get_free_gpu_indices()
scripts = [
    "./exp-synthetic/run_experiment.py",
    "./exp-adni/run_experiment.py",
]

if len(available_gpus) >= 2:
    print("run experiments in parallel.")  # noqa
    return_codes = [run_script(script, available_gpus[i]).wait() for i, script in enumerate(scripts)]
else:
    print("run experiments sequentially.")  # noqa
    for script in scripts:
        proc = run_script(script)
        proc.wait()
