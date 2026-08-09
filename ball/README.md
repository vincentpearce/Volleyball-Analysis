# Ball tracking (Phase 1)

Model: vball-net (TrackNetV4-based, `third_party/vball-net/`), fine-tuned from the
project's pretrained `VballNetV1` checkpoint. Chosen over a generic YOLOv8n baseline
after evaluation showed YOLO's COCO "sports ball" class essentially can't find this
kind of small/blurred ball at all (~17% recall, mostly false positives).

Current weights: `ball/weights/vballnet_finetuned_v3.onnx`, fine-tuned on ~1180
hand-labeled frames spanning two venues/rallies. Fine-tuning traded recall for
precision: 0% wildly-wrong detections (was ~45% on the worst venue pre-fine-tune),
but recall dropped to ~62-95% depending on venue. Lean on `ball.max_gap_interpolation_frames`
in `config/config.yaml` to bridge the resulting gaps.

## Adding more labeled data / re-training

1. Trim a clip: `ffmpeg -ss <timestamp> -i <source> -t 30 -c:v libx264 -crf 18 data/samples/<name>.mp4`
2. Label it (GUI, left-click = ball, middle-click = not visible, `n`/`p` to step,
   `s` to save, `e` to exit):
   ```
   .venv-tf/bin/python third_party/vball-net/src/imgLabel.py --video_path data/samples/<name>.mp4 [--csv_path third_party/vball-net/csv/<name>_ball.csv]
   ```
   Dense, frame-by-frame passes are far more useful than sparse jump-labeling --
   training needs 3+ *consecutive* reviewed frames per window, and the tool can't
   distinguish "confirmed no ball" from "never looked at" in the saved CSV.
3. Build the training data (auto-segments the CSV into one track per contiguous
   labeled run, ≥3 frames, with its own train/test split):
   ```
   .venv/bin/python ball/prepare_vballnet_dataset.py --video_path data/samples/<name>.mp4 --csv_path third_party/vball-net/csv/<name>_ball.csv --track_id <name>
   ```
   Re-run this after every labeling session before training -- it wipes and
   rebuilds `third_party/vball-net/data/frames/{train,test}/`.
4. Reset to the checkpoint you want to fine-tune from (the resume logic picks
   whichever `.keras` file has the latest mtime):
   ```
   cd third_party/vball-net/models/VballNetV1/VballNetV1 && ls *.keras | grep -v _150.keras | xargs rm -f
   ```
5. Train (adjust `--epochs`; it's pretrained-epoch-count + how many more you want,
   e.g. 150 + 10 = `--epochs 160`):
   ```
   cd third_party/vball-net && ../../.venv-tf/bin/python src/train_v1.py --resume --model_name VballNetV1 --epochs 160
   ```
   **Watch memory** (`vm_stat`) during long runs -- this has twice hit a severe
   slowdown (step time going from ~2s to 40-60s) near the end of a run, likely a
   TF/Metal memory leak accumulating over many epochs. If free pages crater and
   CPU% on the process drops, kill it -- a checkpoint is saved every epoch, so
   you lose at most one epoch's progress. Root cause not yet diagnosed.
6. Convert the checkpoint you want to keep, and re-evaluate before trusting it:
   ```
   .venv-tf/bin/python src/keras2onnx.py --model_path models/VballNetV1/VballNetV1/VballNetV1_<epoch>.keras
   ```
   Then run `src/inference_onnx.py` on a labeled clip and compare predictions
   against the ground-truth CSV (recall + pixel error), not just a visual
   spot-check of a couple of frames -- a full-frame screenshot is easy to
   misread (a ceiling light and a ball both look like a small round blob at
   1920x1080 scaled down).

## Patches applied to the vendored `third_party/vball-net` repo

- `src/imgLabel.py`: removed a leftover `pdb.set_trace()` that froze the tool on
  launch; the idle loop was re-decoding the current frame from disk on every
  polling tick (hundreds of times/sec), which appears to have starved mouse-click
  event delivery -- clicks now trigger an immediate redraw instead.
- `src/train_v1.py`: `--resume` used `tf.keras.models.load_model()`, which can't
  deserialize the model's custom layers (`MotionPromptLayer` etc.) since they
  aren't registered as serializable. Changed to build the model in code and
  `load_weights()` from the checkpoint instead.
