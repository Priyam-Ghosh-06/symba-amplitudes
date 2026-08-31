set -x
python -u scripts/run_experiment.py --theory QCD --protocol record --seeds 0 \
  --arms full_vanilla_dense math_only graph_only full_vanilla_moe \
  --epochs 120 --patience 25 --eval-every 10 --batch-size 4 --skip-baselines \
  --out results/QCD_record_long_seed0.json
