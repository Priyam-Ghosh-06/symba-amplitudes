set -x
python scripts/run_experiment.py --theory QCD --protocol record --seeds 0 \
  --arms full_vanilla_dense math_only graph_only full_xsa_proj_dense \
  --epochs 30 --patience 9 --eval-every 3 --batch-size 4 --out results/QCD_record_seed0.json
python scripts/run_experiment.py --theory QCD --protocol template --seeds 0 \
  --arms full_vanilla_dense \
  --epochs 30 --patience 9 --eval-every 3 --batch-size 4 --out results/QCD_template_seed0.json
