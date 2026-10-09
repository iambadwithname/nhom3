import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full", auto_download=["html"])


@app.cell
def _():
    import kagglehub
    import torch
    from torch import nn
    import torch.nn.functional as F
    import polars as pl
    from torchvision.io import read_image
    import matplotlib.pyplot as plt
    from numba import jit
    import os
    from torchvision.datasets import ImageFolder
    from torchvision import datasets 
    from torchvision import transforms
    from torchvision.transforms import ToTensor
    import torchvision
    import lib.my
    from itertools import islice
    from torch.utils.data import DataLoader
    import marimo as mo

    return (
        DataLoader,
        datasets,
        islice,
        jit,
        kagglehub,
        lib,
        mo,
        os,
        plt,
        read_image,
        torch,
        torchvision,
        transforms,
    )


@app.cell
def _(datasets, kagglehub, lib, torchvision, transforms):
    ##download data
    datapath = kagglehub.dataset_download("sachinkumar413/covid-pneumonia-normal-chest-xray-images")

    normalize=   transforms.Normalize(
            mean=0.5067335367202759,
            std=0.23598085343837738,
    )
    normalizei=   transforms.Normalize(
            mean=-normalize.mean*4.2,
            std=1/normalize.std,
    )
    transformi=transforms.Compose([
        normalizei,
        torchvision.transforms.ToPILImage(),
    ])
    transform = transforms.Compose([
        transforms.CenterCrop((232, 232)),
        transforms.ToTensor(),
        transforms.Lambda(lib.my._x),
        normalize
    ])
    #[ "0.5067335367202759", "0.23598085343837738" ]
    #transforms.Normalize( mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), )
    dataset = datasets.ImageFolder(datapath,transform=transform)
    return dataset, transformi


@app.cell
def _(DataLoader, dataset, islice, torch):
    def _f():
        if 0:
            loader = DataLoader(dataset, batch_size=64, shuffle=True,num_workers=20)
            a=torch.tensor([[i[0].mean(),i[0].var()] for i in islice(loader,2000) ])
            return (a[:,0].mean(),(a[:,1].mean()+a[:,0].var()).sqrt())
    _f()
    return


@app.cell
def _(dataset):
    (dataset[0][0].permute(2,0,1))
    return


@app.cell(hide_code=True)
def _(dataset, mo, transformi):
    mo.hstack([
    transformi(dataset[j][0].permute(2,0,1)) for j in range(4)])
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(os):
    for _i in os.walk(f"../../config"):
        print(_i)
    return


@app.cell
def _(path1, plt, read_image):
    plt.imshow(read_image(f"{path1}/COVID/COVID_43.png").permute(1, 2, 0)[:, :, 3:4])
    return


@app.cell
def _(path1, read_image):
    a = read_image(f"{path1}/COVID/COVID_43.png")
    a.permute(1, 2, 0)
    return


@app.cell
def _(path1, read_image, torch):
    images = torch.stack([read_image(f"{path1}/COVID/COVID{f'_{i + 1}' if i else ''}.png") for i in range(1626)]).to(dtype=float)
    return


@app.cell
def _(jit, read_image, torch):
    @jit
    def f():
        images = torch.stack([read_image(f"train/{i}.png") for i in range(1, 50000 + 1)]).to(dtype=float)

    return


@app.cell
def _(os, path1):
    os(path1)
    return


@app.cell
def _():
    print("Asfd")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
