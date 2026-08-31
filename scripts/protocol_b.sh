set -x
python -u scripts/run_experiment.py --theory QED --protocol template --seeds 0 \
  --val-frac 0.15 --test-frac 0.30 \
  --arms full_vanilla_dense math_only graph_only \
  --epochs 30 --patience 12 --eval-every 4 --out results/QED_templateB_seed0.json
python -u scripts/run_experiment.py --theory QCD --protocol template --seeds 0 \
  --val-frac 0.18 --test-frac 0.30 --batch-size 4 \
  --arms full_vanilla_dense \
  --epochs 30 --patience 9 --eval-every 4 --out results/QCD_templateB_seed0.json
