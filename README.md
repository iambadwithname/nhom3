# nhom3 - Chest X-ray Classification

Classify chest X-ray images into 3 classes: COVID-19, Normal, Pneumonia.

Dataset: https://www.kaggle.com/datasets/sachinkumar413/covid-pneumonia-normal-chest-xray-images

## Setup

Install [uv](https://docs.astral.sh/uv/), then:

```
uv sync
```

## Run

```
uv run nhom3 split     # clean + split data
uv run nhom3 tune      # tune lr / batch size
uv run nhom3 train     # train + test all models
uv run nhom3 figures   # plots and tables
```

Code is in `src/nhom3/`, results in `results/`.
