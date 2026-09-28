#!/usr/bin/env bash
# After scripts/rent_wiki.sh: the Wikipedia anneal again, on the full set of articles.
# The Hugging Face dump lacks about a third of them (mostly the most famous), so
# "web.wikipedia fetch" gets those from the API. This runs the fetch at once (network,
# not GPU), then waits for rent_wiki.sh to finish before it uses the GPU.
#
#   nohup scripts/rent_wikifull.sh > logs/web/rent_full.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
LOG=logs/web
STATUS=$LOG/rent_status.txt
W=checkpoints/web
CUDA=$(python -c "import torch; print(int(torch.cuda.is_available()))")
if [ "$CUDA" = 1 ]; then COMPILE=--compile; else COMPILE=; fi

say() { echo "$(date +%H:%M:%S) $*" | tee -a "$STATUS"; }
run() {     # run NAME COMMAND...
    local name=$1; shift
    say "$name: started"
    if "$@" > "$LOG/rent_$name.log" 2>&1; then say "$name: ok"; else say "$name: FAILED, see $LOG/rent_$name.log"; fi
}

[ -e data/vital_articles_4_api.parquet ] || run wiki_fetch python -m web.wikipedia fetch
run wiki_merge python -m web.wikipedia merge
run wiki_encode_full python -m web.wikipedia encode --full

until grep -q "all steps done" "$STATUS"; do sleep 30; done      # the GPU is free
run anneal_wiki25full python -m web.anneal --data wikifull --share 0.25 --tag wiki25full --tokens 1.5e8 $COMPILE
run probe_wiki25full python -m tools.knowledge $W/web_gpt_768w12l_wiki25full.pt --wiki-split --answers
say "full set done. To copy back: $W/*wiki25full* $LOG/rent_*"
