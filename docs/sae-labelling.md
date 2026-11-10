# Labelling SAE features

Every feature of an SAE can be given a short name: what it reacts to in the text, not
what it does for the model. The names live in `<sae>.labels.json` next to the SAE, one
object from feature number to name, so they survive retraining and can be edited by hand.

There are about 6,000 features per layer and the shortlist worth naming is 1,000–2,500
long, so a layer is several sittings. The work is split by *pass*: a pass is a range of
the shortlist handed to one person or agent, and it writes its answers to a file of its
own in `staging/`, next to the labels. Nothing is agreed until a pass is merged.

    python -m tools.sae slice  <sae> --start 40 --n 80     # write features 40-119 for a pass
    (hand the .work.txt to whoever is naming them)
    python -m tools.sae merge  <sae>                       # fold every answer in, report conflicts
    python -m tools.sae merge  <sae> --name features40-119 --force

    python -m tools.sae audit  <sae>                       # each name beside fresh contexts
    python -m tools.sae report <sae>                       # what the check said

`slice` writes `<sae>.<pass>.work.txt`: the features, their top contexts, their firing
rates, and the exact path to write names to. That file is the whole briefing — the style
guide is in its header — so a pass needs nothing else and cannot lose it.

`merge` takes every `<sae>.*.labels.json` in `staging/` (or the passes named with
`--name`). A feature that two passes named differently, or that already has a name, is a
conflict: nothing is written for it and the two names are printed. That is the point of
staging — where agents disagree, a person decides. `--force` takes the first pass in name
order instead. Merging twice is harmless.

`audit` is the other direction: it shows each name already written beside the top
contexts for the same feature in *different* text at a different seed, and asks for a
verdict. It catches names that only fit the eight sentences they were written from, which
is the failure mode of a fast labelling pass and of your own tired guesses. `report`
prints the names that did not survive, and drops a row once the name has been changed
since the check (the `was` field in the verdict), so the list drains instead of filling
with things already dealt with.

Everything in `staging/` is disposable. The labels file is the only thing that matters.

## Prompt: a naming pass

> You are naming sparse-autoencoder features from a small language model trained on
> children's stories. One file has everything you need.
>
> Read `<staging>/<sae>.<pass>.work.txt`. It lists the features, and for each one: how
> strongly it fires at its peak, how often it fires over about 500,000 tokens of stories,
> and its 8 strongest contexts. A context is 24 tokens before, the token it fired on in
> [brackets], and 3 after.
>
> Then write one JSON object from feature number to name, to exactly the path the header
> gives (`<sae>.<pass>.labels.json`). Nothing else goes in that file, and nothing else in
> the folder may change: not the `.pt`, not the `.examples.json`, not the `.labels.json`.
>
> How to name, as the header says: a short lowercase phrase for what the feature reacts
> to, not what it does for the model — "hyphenated compounds", "dialogue verbs", "animal
> words", "the token 'ing'". How often it fires is part of the answer: one on 30% of
> tokens is doing something broad, and a name that only fits eight unusual sentences is
> wrong. Where the contexts genuinely disagree, write `unclear (a or b)` instead of
> inventing a compromise. Leave a feature out only if you cannot name it at all.
>
> Read the names already written for this layer (`<sae>.labels.json`) first, so yours
> match their voice. You are naming only the features in your file.
>
> When done, reply with the number you named, the features you left out, and up to five
> names you are least sure of, one line each on why.

## Prompt: a check pass

> You are checking names already given to sparse-autoencoder features. The work files
> give each name beside 8 *fresh* contexts: the same features found in different text at
> a different seed, so a name that only fits the sentences it was written from will not
> fit these.
>
> Read every `<staging>/<sae>.audit.work.txt` you are given. For each name, judge the
> name as written, then write one JSON object per layer to the `.verdicts.json` path its
> header names, shaped as the header shows:
> `{"12": {"verdict": "agrees"}, "13": {"verdict": "disagrees", "was": "past tense", "label": "past-tense verbs"}}`.
> `verdict` is `agrees`, `partly` or `disagrees`. Give `label` with anything but agrees:
> the name you would write instead, in the same style. Give `was`: the name that was
> checked, so a row whose name has since changed can be told from one that still stands.
>
> Judge as written, not against a better name: "questions" agrees if these are questions,
> even if "wh- questions" is more precise. Weigh the firing rate, not only the contexts: a
> rare feature that fires nowhere in this text is consistent with a rare name. Change
> nothing else in the folder.
>
> Reply with totals per layer, then the ten most wrong names, one line each.

## Prompt: a concept index

> Read `<staging>/../<sae>.labels.json` for each of the six layers. Each maps feature
> number to name, and the same concept turns up in more than one layer.
>
> Write `<staging>/concept-index.md` with three sections. First, concepts that appear in
> more than one layer: the name as used, which layers, which feature numbers; group only
> things that are genuinely the same idea, not words that merely look similar. Second,
> naming inconsistencies: one idea written two ways, or two ideas sharing a name, with one
> form proposed for each. Third, at most ten one-line observations that the labels
> support, such as a concept many layers share or one a single layer has.
>
> Use only the labels and the feature numbers. Where names are too vague to group, say so
> instead of guessing. Keep it under 150 lines, and change none of the `.labels.json`.

## What to expect

The head of the shortlist is distinctive (orthographic features in the early layers,
grammatical and narrative ones in the later ones) and the tail is not: past a few hundred
features the honest label is a sentence, not a name. So passes are capped at 60–100
features and the later layers (3–5) are the ones worth spending on. About 0–1 in 100
shortlisted features duplicate another's top context, so passes can be split by range
without two of them naming the same thing.
