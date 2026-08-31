set -x
python -u scripts/run_experiment.py --theory QED --protocol record --seeds 0 \
  --arms full_vanilla_dense math_only graph_only full_xsa_proj_dense full_xsa_mask_dense full_vanilla_moe unconstrained_decode capacity_256 \
  --epochs 30 --patience 12 --eval-every 5 --out results/QED_record_seed0.json
python -u scripts/run_experiment.py --theory QED --protocol record --seeds 1 2 \
  --arms full_vanilla_dense math_only graph_only \
  --epochs 30 --patience 12 --eval-every 5 --out results/QED_record_seeds12.json
