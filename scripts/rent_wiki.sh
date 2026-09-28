#!/usr/bin/env bash
# One rental on a 5090: the Wikipedia anneal and the jobs that go with it.
#
# Run from the repo root, after cloning, with these uploaded (see the README):
#   checkpoints/web/web_bpe.json                  the web tokeniser
#   checkpoints/web/web_gpt_768w12l.pt            the final web model: base of the Wikipedia anneal
#   checkpoints/web/web_gpt_768w12l_anneal25.pt   base of the v2 system SFT
#   checkpoints/web/web_gpt_768w12l_anneal.pt     base of the longer instruct SFT (EXTRAS only)
#   data/vital_articles_4.json                    optional: else "list" makes it again (~2 min)
#
#   nohup scripts/rent_wiki.sh > logs/web/rent.log 2>&1 < /dev/null &
#   tail -f logs/web/rent_status.txt              # one line per step: ok, FAILED or skipped
#
# Each step logs to logs/web/rent_<step>.log. A step whose output exists is skipped, so
# after a problem, run the script again. A failed step doesn't stop the others.
# Settings, as environment variables (the smoke test on the Mac makes them tiny):
WIKI_FILES=${WIKI_FILES:-41}                  # Wikipedia dump files, of 41
WEB_TRAIN_FILE=${WEB_TRAIN_FILE:-000}         # the FineWeb-Edu file for the web text in the mix
WEB_TOKENS=${WEB_TOKENS:-4e8}                 # enough web text for a 25% mix of ~105M wiki tokens
ANNEAL_TOKENS=${ANNEAL_TOKENS:-1.5e8}
ANNEAL_EXTRA=${ANNEAL_EXTRA:-}                # e.g. "--batch 4 --accum 1 --val-batch 4" on the Mac
CHAT_STEPS=${CHAT_STEPS:-250}
SYSTEM_STEPS=${SYSTEM_STEPS:-300}
SFT_TESTS=${SFT_TESTS:-20}
INSTRUCT_STEPS=${INSTRUCT_STEPS:-12000}
ABLATE=${ABLATE:-1}                           # 0: skip the 144-head ablation
EXTRAS=${EXTRAS:-1}                           # 0: skip the 50% anneal and the longer instruct SFT

set -u
cd "$(dirname "$0")/.."
W=checkpoints/web
LOG=logs/web
STATUS=$LOG/rent_status.txt
mkdir -p "$LOG" data "$W"
CUDA=$(python -c "import torch; print(int(torch.cuda.is_available()))")
if [ "$CUDA" = 1 ]; then COMPILE=--compile; ACCUM=1; else COMPILE=; ACCUM=4; fi
HF=https://huggingface.co/datasets

step() {    # step NAME OUTPUT COMMAND...: run COMMAND, unless OUTPUT exists ("" = always run)
    local name=$1 out=$2; shift 2
    if [ -n "$out" ] && [ -e "$out" ]; then
        echo "$(date +%H:%M:%S) $name: skipped, $out exists" | tee -a "$STATUS"; return
    fi
    echo "$(date +%H:%M:%S) $name: started" | tee -a "$STATUS"
    if "$@" > "$LOG/rent_$name.log" 2>&1; then
        echo "$(date +%H:%M:%S) $name: ok" | tee -a "$STATUS"
    else
        echo "$(date +%H:%M:%S) $name: FAILED, see $LOG/rent_$name.log" | tee -a "$STATUS"
    fi
}

fetch() {   # fetch URL PATH: download to PATH.tmp, then rename, so a half file never counts as done
    curl -sSfL -o "$2.tmp" "$1" && mv "$2.tmp" "$2"
}

# ---------- data ----------
step fineweb_val_file data/fineweb_edu_013.parquet \
    fetch $HF/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/013_00000.parquet data/fineweb_edu_013.parquet
step web_val data/web_val.bin python -m web.web_data encode val
step fineweb_train_file data/fineweb_edu_$WEB_TRAIN_FILE.parquet \
    fetch $HF/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/${WEB_TRAIN_FILE}_00000.parquet \
    data/fineweb_edu_$WEB_TRAIN_FILE.parquet
step web_train data/web_train.bin \
    python -m web.web_data encode train data/fineweb_edu_$WEB_TRAIN_FILE.parquet --max-tokens "$WEB_TOKENS"
step wiki_list data/vital_articles_4.json python -m web.wikipedia list
step wiki_download "" python -m web.wikipedia download --files "$WIKI_FILES"      # skips files it has
step wiki_filter data/vital_articles_4.parquet python -m web.wikipedia filter --files "$WIKI_FILES"
step wiki_encode data/wiki_train.bin python -m web.wikipedia encode
step dolly data/dolly-15k.jsonl \
    fetch $HF/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl data/dolly-15k.jsonl

# ---------- the Wikipedia anneal, and what it changed ----------
step probe_before "" python -m tools.knowledge $W/web_gpt_768w12l.pt --wiki-split --answers
step anneal_wiki25 $W/web_gpt_768w12l_wiki25.pt \
    python -m web.anneal --data wiki --share 0.25 --tag wiki25 --tokens "$ANNEAL_TOKENS" $COMPILE $ANNEAL_EXTRA
step probe_wiki25 "" python -m tools.knowledge $W/web_gpt_768w12l_wiki25.pt --wiki-split --answers
step chat_wiki25 $W/web_gpt_768w12l_wiki25_chat_s250.pt \
    python -m web.chat_sft --base $W/web_gpt_768w12l_wiki25.pt --steps "$CHAT_STEPS" --accum $ACCUM --tag s250
step probe_chat_wiki25 "" python -m tools.knowledge $W/web_gpt_768w12l_wiki25_chat_s250.pt --wiki-split

# ---------- the other jobs ----------
if [ "$ABLATE" = 1 ]; then
    step ablate "" python -m tools.interp --checkpoint $W/web_gpt_768w12l.pt --ablate
fi
step system_v2 $W/web_gpt_768w12l_anneal25_system_v2.pt \
    python -m web.system_sft --tag v2 --steps "$SYSTEM_STEPS" --accum $ACCUM --tests "$SFT_TESTS"

if [ "$EXTRAS" = 1 ]; then
    step anneal_wiki50 $W/web_gpt_768w12l_wiki50.pt \
        python -m web.anneal --data wiki --share 0.5 --tag wiki50 --tokens "$ANNEAL_TOKENS" $COMPILE $ANNEAL_EXTRA
    step probe_wiki50 "" python -m tools.knowledge $W/web_gpt_768w12l_wiki50.pt --wiki-split --answers
    step instruct_val data/TinyStories-Instruct-valid.txt \
        fetch $HF/roneneldan/TinyStoriesInstruct/resolve/main/TinyStories-Instruct-valid.txt \
        data/TinyStories-Instruct-valid.txt
    step instruct_train data/TinyStories-Instruct-train-300MB.txt bash -c \
        "curl -sSL $HF/roneneldan/TinyStoriesInstruct/resolve/main/TinyStories-Instruct-train.txt \
         | head -c 314572800 > data/instruct.tmp && mv data/instruct.tmp data/TinyStories-Instruct-train-300MB.txt"
    step instruct_12k $W/stories_instruct_web_gpt_768w12l_anneal_300mb_12k.pt \
        python -m stories.instruct_sft --base $W/web_gpt_768w12l_anneal.pt \
        --train-file data/TinyStories-Instruct-train-300MB.txt --steps "$INSTRUCT_STEPS" --lr 2e-4 --tag 300mb_12k
fi

echo "$(date +%H:%M:%S) all steps done. To copy back: $W/*wiki* $W/*system_v2* $W/*12k* $LOG/rent_*" \
    | tee -a "$STATUS"
