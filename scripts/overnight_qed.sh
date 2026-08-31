set -x
python scripts/run_experiment.py --theory QED --protocol record --seeds 0 \
  --arms full_vanilla_dense math_only graph_only full_xsa_proj_dense full_xsa_mask_dense full_vanilla_moe capacity_256 unconstrained_decode \
  --epochs 40 --patience 12 --eval-every 3 --out results/QED_record_seed0.json
python scripts/run_experiment.py --theory QED --protocol template --seeds 0 \
  --arms full_vanilla_dense math_only graph_only \
  --epochs 40 --patience 12 --eval-every 3 --out results/QED_template_seed0.json
python scripts/run_experiment.py --theory QED --protocol record --seeds 1 2 \
  --arms full_vanilla_dense full_xsa_proj_dense math_only graph_only \
  --epochs 40 --patience 12 --eval-every 3 --out results/QED_record_seeds12.json
