conda_path="$1"
conda_env="$2"

niters=200
test_split=0.2

$conda_path run -n $conda_env python asac_baseline.py --niters=$niters --test-split=$test_split --seed=0
$conda_path run -n $conda_env python asac_baseline.py --niters=$niters --test-split=$test_split --seed=1
$conda_path run -n $conda_env python asac_baseline.py --niters=$niters --test-split=$test_split --seed=2
$conda_path run -n $conda_env python asac_baseline.py --niters=$niters --test-split=$test_split --seed=3
$conda_path run -n $conda_env python asac_baseline.py --niters=$niters --test-split=$test_split --seed=4
