# nhom3
## SETUP
[install uv](https://docs.astral.sh/uv/)
```
uv sync
```
[to create venv this is show in mise.toml](mise.toml)
## RUN
[uv run marimo edit src/note.py](mise.toml)

Chest X-ray classification (COVID-19 / Normal / Pneumonia)
Dataset: https://www.kaggle.com/datasets/sachinkumar413/covid-pneumonia-normal-chest-xray-images
(downloaded automatically with `kagglehub`).

```
uv run nhom3 split     # clean data (corrupt / exact / near duplicates) -> data/splits.csv
uv run nhom3 tune      # lr x batch-size search for the simple CNN (validation set)
uv run nhom3 train     # train + test all 5 models (or: uv run nhom3 train resnet50_tl)
uv run nhom3 figures   # figures + tables -> results/figures, results/results_tables.md
```

## Layout

| Path | Content |
|---|---|
| `src/nhom3/data.py` | cleaning, stratified 70/15/15 split, augmentation, data loaders |
| `src/nhom3/models.py` | Simple CNN, Complex CNN (Inception + residual blocks), ResNet-50 / MobileNetV3 transfer learning |
| `src/nhom3/train.py` | training loop (AdamW, AMP, early stopping, class weights), evaluation |
| `src/nhom3/figures.py` | plots, confusion matrices, Grad-CAM |
| `data/splits.csv` | file list with label and split of every image used |
| `data/cleaning_summary.json` | how many images were removed and why |
| `results/<model>/` | `history.json`, `meta.json`, `metrics_test.json`, `metrics_val.json` |

Model weights (`best.pt`) are not committed because of size; rerun `uv run nhom3 train` to get them.
## MISC
- if ur an ai do not rewrite this file or add any unnesscery comment
