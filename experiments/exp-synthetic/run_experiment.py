import argparse
import sys
from pathlib import Path

# Add repo root to PYTHONPATH to ensuer we can access experiments module:
sys.path.append((Path(__file__).parent.parent.parent).as_posix())

from experiments.exp_config import conda_path, epochs, venv, venv_asac
from experiments.exp_helper import get_cmd, run_script

parser = argparse.ArgumentParser("Experiment on synthetic dataset")
parser.add_argument("--gpu", type=int, default=0)
args = parser.parse_args()
gpu = args.gpu


print("Obtain the baseline predictor...")  # noqa
# First find the best predictor.
check_result = Path("./predictor/best_drop_rate")
if check_result.exists():
    print("The baseline predictor is already obtained. Skip.")  # noqa: T201
else:
    # Epochs can be set to 100 since we do early stopping and the training stops soon.
    cmd = get_cmd("./obtain_predictor.py", venv=venv, gpu=gpu, epochs=100)
    proc = run_script(cmd, "exp-synthetic")
    proc.wait()
    # The results should be written to Path("predictor")
    if not check_result.exists():
        raise RuntimeError("Error: Failed to obtain the baseline predictor.")

print("Evaluate RAS...")  # noqa
# Then we train the ras model.
lambda_list = [200.0, 250.0, 280.0, 300.0, 310.0, 320.0, 350.0, 400.0]
for λ in lambda_list:
    check_result = Path(f"./ras/lambda={λ}.csv")
    if check_result.exists():
        print(f"RAS(λ={λ}) is already trained. Skip.")  # noqa: T201
        continue

    param = {"coeff": λ}
    cmd = get_cmd("./exp_ras.py", venv=venv, gpu=gpu, epochs=epochs, parameters=param)
    proc = run_script(cmd, "exp-synthetic")
    proc.wait()

    if not check_result.exists():
        raise RuntimeError(f"Error: Failed to train model RAS(λ={λ}). Abort.")


print("Evaluate NLL...")  # noqa
# Then we train the nll model.
lambda_list = [100.0, 300.0]
for λ in lambda_list:
    check_result = Path(f"./nll/lambda={λ}.csv")
    if check_result.exists():
        print(f"NLL(λ={λ}) is already trained. Skip.")  # noqa: T201
        continue

    param = {"coeff": λ}
    cmd = get_cmd("./exp_nll.py", venv=venv, gpu=gpu, epochs=epochs, parameters=param)
    proc = run_script(cmd, "exp-synthetic")
    proc.wait()

    if not check_result.exists():
        raise RuntimeError(f"Error: Failed to train model NLL(λ={λ}). Abort.")

print("Evaluate AS...")  # noqa
# Then we train the baseline (AS) model.
delta_list = [0.2, 0.5, 1.0]
for δ in delta_list:
    check_result = Path(f"./baseline/delta={δ}.csv")
    if check_result.exists():
        print(f"AS(delta={δ}) is already trained. Skip.")  # noqa: T201
        continue

    param = {"delta": δ}
    cmd = get_cmd("./exp_baseline.py", venv=venv, gpu=gpu, epochs=epochs, parameters=param)
    proc = run_script(cmd, "exp-synthetic")
    proc.wait()

    if not check_result.exists():
        raise RuntimeError(f"Error: Failed to train model AS(delta={δ}). Abort.")


print("Evaluate ASAC...")  # noqa
# Finally, we train the ASAC baseline.
# Note that the result of the ASAC baseline is always random and we have no way to resovle this issue.
cmd = get_cmd("./exp_asac.py", venv=venv_asac, gpu=None, epochs=None)
proc = run_script(cmd, "exp-synthetic")
proc.wait()

print("Evaluate sensing deficiency...")  # noqa
# Do a small ablation study
cmd = get_cmd("./exp_ablation.py", venv=venv, gpu=gpu, epochs=None)
proc = run_script(cmd, "exp-synthetic")
proc.wait()


print("Perform benchmark...")  # noqa
cmd = get_cmd("./evaluation.py", venv=venv, gpu=gpu, epochs=None)
proc = run_script(cmd, "exp-synthetic")
proc.wait()
