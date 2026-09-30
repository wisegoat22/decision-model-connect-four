# Re-render the recorded highlights

This renders existing game records and verifies each block against the board. It does not load models, create moves or change results. The selection was made after inspecting outcomes for visible defensive play and length; it is not a representative sample.

The exported files were checked on macOS with Pillow 12.3.0 and imageio-ffmpeg 0.6.0. Avenir Next is used on macOS; DejaVu Sans is the fallback on Linux. A custom TrueType font can be supplied with `--font /path/to/font.ttf`. Fonts are not bundled, and different fonts can change the appearance.

From the repository root, using a separate environment if desired:

```sh
python3 -m pip install -r video/requirements.txt
python3 video/render_highlights.py --data evidence/results.json --selection media/highlights/selection.json --output runs/highlights --review-dir runs/highlight-review
```

Add `--preview-only` to produce source-frame previews without movies. The renderer refuses to overwrite an existing movie. Keep new outputs in `runs/`, leaving the published files and original evidence unchanged.

The renderer verifies all 60 recorded games, the selected source-file hashes, every annotated immediate-threat block, and the initial and final boards. Each final movie is fully decoded, sampled decoded frames are saved for human inspection, and media properties are recorded. The script leaves visual review pending until those images are actually inspected. The committed `media/highlights/QA.json` documents that completed review for the published exports.

The excerpt starts from the recorded board after move 14 and includes all moves 15–24. The two seeded full games start after the four preset opening pieces; those pieces are explicitly labeled and never animated as model choices. All displayed model moves use one second between move starts, with a brief initial and final hold.
