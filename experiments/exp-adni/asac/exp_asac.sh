eval "$(conda shell.bash hook)"
conda activate asac_sensing

niters=200
test_split=0.2

python asac_baseline.py --niters=$niters --test-split=$test_split --seed=0
python asac_baseline.py --niters=$niters --test-split=$test_split --seed=1
python asac_baseline.py --niters=$niters --test-split=$test_split --seed=2

conda deactivate
