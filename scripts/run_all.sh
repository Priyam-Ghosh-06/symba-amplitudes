#!/usr/bin/env bash
# Train every model in the results tables, three at a time, then draw the figures.
#
#   bash scripts/run_all.sh
#
# A model whose results/<theory>_<variant>.json already exists is skipped, so an
# interrupted run picks up where it stopped. Logs go to logs/.
cd "$(dirname "$0")/.."
mkdir -p logs

# Encoder ablations on both theories; component ablations on QED, where a run
# is four times cheaper.
printf '%s\n' \
    "QCD full" "QCD ast_only" "QED full" "QCD graph_only" "QED ast_only" "QED graph_only" \
    "QED no_moe" "QED no_role_filler" "QED no_xsa" |
xargs -P 3 -L 1 sh -c '
    [ -f "results/$0_$1.json" ] && exit 0
    if python scripts/train.py --theory "$0" --variant "$1" --threads 4 > "logs/$0_$1.log" 2>&1
    then echo "done    $0 $1"
    else echo "FAILED  $0 $1 (see logs/$0_$1.log)"
    fi'

python scripts/plot_results.py
