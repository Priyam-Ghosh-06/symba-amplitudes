set -x
python -u scripts/run_experiment.py --theory QED --protocol record --seeds 0 \
  --arms full_vanilla_dense math_only graph_only \
  --epochs 120 --patience 25 --eval-every 10 --skip-baselines \
  --out results/QED_record_long_seed0.json
