# nhom3 - Chest X-ray Classification

Classify chest X-ray images into 3 classes: COVID-19, Normal, Pneumonia.

Dataset: https://www.kaggle.com/datasets/sachinkumar413/covid-pneumonia-normal-chest-xray-images

## Setup

Install [uv](https://docs.astral.sh/uv/), then:

```
uv sync
```

## Run

Run the files in order (each file trains and saves its models to `results/<model>/best.pt`):

```
uv run python src/1_preprocess.py        # clean + split data
uv run python src/2_simple_cnn.py        # simple CNN (tune lr / batch size, then train)
uv run python src/3_complex_cnn.py       # complex CNN
uv run python src/4_transfer_learning.py # MobileNetV3 (transfer learning)
uv run python src/6_figures.py           # plots and tables
```

Use the saved models without training again:

```
uv run python src/5_predict.py                  # predict the test set
uv run python src/5_predict.py path/to/image.png # predict one image
```

The same code (steps 1-6) is also written as one marimo notebook:

```
uv run marimo edit src/notebook.py
```

Shared code (data loading, training loop, evaluation) is in `src/utils.py`, models in `src/models.py`.
