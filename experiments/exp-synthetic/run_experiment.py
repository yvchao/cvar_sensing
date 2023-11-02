import subprocess
from pathlib import Path

epochs = 200
gpu = 0
venv = "cvar_sensing"


def get_cmd(script: Path | str, gpu: int = 0, epochs: int = 200, parameters: dict[str, int | float] = {}):
    script = Path(script)
    cmd = [
        f"conda run -n {venv} python".split(" "),
        script.as_posix(),
        "--gpu",
        f"{gpu}",
        "--epochs",
        f"{epochs}",
    ]
    for k, v in parameters.items():
        cmd += [f"--{k}", f"{v}"]

    return cmd


# First find the best predictor.
check_result = Path("./predictor/best_model_id")
if check_result.exists():
    print("The baseline predictor is already obtained. Skip.")  # noqa: T201
else:
    # Epochs can be set to 100 since we do early stopping and the training stops soon.
    cmd = get_cmd("./obtain_predictor.py", gpu=gpu, epochs=100)
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    proc.wait()
    # The results should be written to Path("predictor")
    if not check_result.exists():
        raise RuntimeError("Error: Failed to obtain the baseline predictor.")


# Then we train the ras model.
lambda_list = [200.0, 250.0, 280.0, 300.0, 310.0, 320.0, 350.0, 400.0]
for λ in lambda_list:
    check_result = Path(f"./ras/lambda={λ}.csv")
    if check_result.exists():
        print(f"RAS(λ={λ}) is already trained. Skip.")  # noqa: T201
        continue

    param = {"coeff": λ}
    cmd = get_cmd("./exp_ras.py", gpu=gpu, epochs=epochs, parameters=param)
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    proc.wait()

    if not check_result.exists():
        raise RuntimeError(f"Error: Failed to train model RAS(λ={λ}). Abort.")


# Then we train the nll model.
lambda_list = [100.0, 300.0]
for λ in lambda_list:
    check_result = Path(f"./nll/lambda={λ}.csv")
    if check_result.exists():
        print(f"NLL(λ={λ}) is already trained. Skip.")  # noqa: T201
        continue

    param = {"coeff": λ}
    cmd = get_cmd("./exp_nll.py", gpu=gpu, epochs=epochs, parameters=param)
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    proc.wait()

    if not check_result.exists():
        raise RuntimeError(f"Error: Failed to train model NLL(λ={λ}). Abort.")

# Then we train the baseline (AS) model.
delta_list = [0.2, 0.5, 1.0]
for δ in delta_list:
    check_result = Path(f"./baseline/delta={δ}.csv")
    if check_result.exists():
        print(f"AS(delta={δ}) is already trained. Skip.")  # noqa: T201
        continue

    param = {"delta": δ}
    cmd = get_cmd("./exp_baseline.py", gpu=gpu, epochs=epochs, parameters=param)
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    proc.wait()

    if not check_result.exists():
        raise RuntimeError(f"Error: Failed to train model AS(delta={δ}). Abort.")


# Finally, we train the ASAC baseline.
# Note that the result of the ASAC baseline is always random and we have no way to resovle this issue.
cmd = f"conda run -n {venv} python ./exp_asac.py"
proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
proc.wait()
