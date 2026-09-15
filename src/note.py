import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full", auto_download=["html"])


@app.cell
def _():
    import kagglehub

    return (kagglehub,)


@app.cell
def _(kagglehub):
    ##download data
    path1 = kagglehub.dataset_download(
        "sachinkumar413/covid-pneumonia-normal-chest-xray-images"
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
