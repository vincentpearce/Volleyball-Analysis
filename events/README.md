# Event timestamping (Phase 3)

Active classifier: **heuristic rules** (`events/classify.py`), per CLAUDE.md's
explicit MVP direction. A ball flight segment (`ball/postprocess.py`'s
`tracked_segments`, further split at real hit points by
`events/touch_detection.py`'s velocity-discontinuity detector) is classified
by rules on displacement/arc-height/jump-correlation. See `classify.py`'s
docstring for the exact rule order.

## Validated, unlike earlier phases claimed -- read this before trusting output

`events/label_events.py` is the ground-truth tool for this phase (plays each
segment back, asks for the real label). Real result on clip 2's 26
confidently-labeled segments: **8/26 (30.8%) correct -- below the 38.5%
majority-class baseline** (always guessing the more common of set/dig would
score higher). The errors aren't random: set/dig get confused in both
directions equally often, meaning `set_min_apex_rise_px` isn't actually
discriminative on this data, and several real spikes get called "set" because
Phase 2's contact detection didn't confirm a jump for them. Full breakdown
was worked through in-session; the honest summary is **this heuristic is not
currently better than guessing the most common class**, though it does
produce *something* structured and running end-to-end.

## Two classifiers: heuristic (default) and learned (for professional footage)

Tried the CLAUDE.md-sanctioned fallback: downloaded a public per-player
action dataset (Roboflow "Volleyball Activity Dataset", Graz Uni -- 17K
labeled player crops, fixed sideline camera, classes matching ours) and
fine-tuned a YOLO11s-cls classifier (`events/prepare_action_dataset.py` to
build the dataset, training command in git history / rerun similarly).
It scored 94.7% on Roboflow's own held-out test set -- and **~11.5% on our
own footage** (`events/evaluate_classifier.py`, checked against the same
human labels above). That's a real domain-gap failure, not a bug in the eval
script (frame-sampling was checked and fixed; accuracy barely moved).

Kept and wired in anyway (`events/learned_classify.py`, select with
`events/run.py --classifier learned`) because it should transfer much better
to footage that actually looks like its training data -- i.e. **professional
or high-level play**, not this project's own club-level sample clips. Use
`--classifier heuristic` (the default) for footage like the samples in this
repo; use `--classifier learned` for more polished match footage. Model
weights: `events/weights/action_classifier.pt` (path configurable via
`config.yaml`'s `events.learned_classifier_path`). Doesn't need `--pose_json`
(no jump-correlation step -- classifies straight from the cropped player
image). If accuracy on professional footage turns out disappointing too, the
next step is fine-tuning this same model on labeled crops from whatever
footage it's actually being pointed at, the same playbook as `ball/`'s
fine-tune -- not re-tuning the heuristic further.

## Footage is casual/club-level play, not professional -- this matters

Our source footage is not high-level play. A meaningful fraction of the
confusing/misclassified instances are likely genuine ambiguity in the
underlying action, not just measurement error: a poorly-executed dig, or a
block/dig that's technically unnecessary against a weak attack, doesn't have
the clean, textbook kinematic signature (full jump, sharp arm swing, crisp
platform) that both the heuristic thresholds and the Roboflow model (trained
on professional Austrian Volley League play) assume. This is a plausible
contributor to the set/dig confusion above and to the learned model's failure
to transfer -- worth keeping in mind before assuming any fix here is purely a
threshold or training-data problem.

## Known gaps in the heuristic (still true, now measured rather than guessed)

- No net-position awareness (no court calibration exists in this project) --
  block vs. spike is distinguished purely by how far the ball travels
  afterward.
- Jump serves would get classified as "serve" (new-rally rule fires first,
  before the jump-contact check).
- Set vs. dig confusion is confirmed real and roughly symmetric (see above),
  not just a theoretical risk.
