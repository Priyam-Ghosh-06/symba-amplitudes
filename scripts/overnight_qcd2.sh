set -x
python -u scripts/run_experiment.py --theory QCD --protocol record --seeds 0 \
  --arms full_vanilla_moe unconstrained_decode capacity_256 \
  --epochs 30 --patience 9 --eval-every 3 --batch-size 4 --out results/QCD_record_seed0_extra.json
python -u scripts/run_experiment.py --theory QCD --protocol record --seeds 1 2 \
  --arms full_vanilla_dense math_only graph_only \
  --epochs 30 --patience 9 --eval-every 3 --batch-size 4 --out results/QCD_record_seeds12.json
