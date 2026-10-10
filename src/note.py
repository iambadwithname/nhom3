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
    from itertools import islice
    from torch.utils.data import DataLoader
    import marimo as mo
    from lib import my2
    from torch.utils.data import random_split, Subset
    from torchinfo import summary
    import marimo as mo
    from lib.my import dataset, transformi
    from torcheval.metrics.aggregation.auc import AUC

    return (
        AUC,
        DataLoader,
        F,
        Subset,
        dataset,
        islice,
        mo,
        my2,
        nn,
        random_split,
        summary,
        torch,
        transformi,
    )


@app.cell
def _():
    # setup
    # torch._logging.set_logs(graph_code=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## DATA
    """)
    return


@app.cell
def _(dataset):
    dataset[0][0].shape
    return


@app.cell
def _(dataset, random_split, torch):
    def _f_split(dataset):
        n = len(dataset)
        l = [int(n * 0.6), int(n * 0.2)]  # why 622 cause duh
        l.append(n - sum(l))
        return random_split(dataset, l, torch.Generator().manual_seed(67))

    dataset_train, dataset_valid, dataset_test = _f_split(dataset)
    return dataset_test, dataset_train, dataset_valid


@app.cell(hide_code=True)
def _(DataLoader, dataset, islice, torch):
    def _f():  # this is to get std and mean of the dataset just ignore it
        if 0:
            loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=20)
            a = torch.tensor([[i[0].mean(), i[0].var()] for i in islice(loader, 2000)])
            return (a[:, 0].mean(), (a[:, 1].mean() + a[:, 0].var()).sqrt())

    _f()
    return


@app.cell(hide_code=True)
def _(dataset, mo, transformi):
    mo.hstack([transformi(dataset[j][0]) for j in range(4)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## SimpleCNN
    """)
    return


@app.cell
def _(dataset_train, mo, nn):
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.drop=nn.Dropout(0.09)
            self.CNN = nn.Sequential(
                nn.Conv2d(1, 2, (3, 3)),
                nn.Conv2d(2, 4, (3, 3), (1, 1), (3, 3)),
                self.drop,
                nn.MaxPool2d((3, 3), (1, 1), (1, 1)),
                nn.Conv2d(4, 8, (3, 3), (3, 3)),
                self.drop,
                nn.MaxPool2d((3, 3), (1, 1), (1, 1)),
                nn.Conv2d(8, 16, (3, 3), (3, 3)),
                self.drop,
                nn.MaxPool2d((3, 3), (1, 1), (1, 1)),
                nn.Conv2d(16, 32, (3, 3), (3, 3)),
                self.drop,
                nn.MaxPool2d((3, 3), (1, 1), (1, 1)),
                nn.Conv2d(32, 64, (3, 3), (3, 3)),
            )
            self.pooling = nn.AdaptiveAvgPool2d((2, 2))
            self.Activation = nn.Sequential(nn.ReLU())
            self.fc = nn.Sequential(
                nn.Linear(256, 64),
                self.Activation,
                nn.Linear(64, 32),
                self.Activation,
                nn.Linear(32, 3),
            )

        def forward(self, x):
            x = self.CNN(x)
            x = self.pooling(x)
            x = x.flatten(-3)
            x = self.fc(x)
            x = x.softmax(-1)
            return x

    model1_buttons = mo.ui.dictionary({"train size": mo.ui.range_slider(0, len(dataset_train), value=(0, 256), full_width=True, label="train size")} | {i: mo.ui.run_button(label=i) for i in ["train", "test", "valid", "load", "save", "reset"]})
    return SimpleCNN, model1_buttons


@app.cell
def _(SimpleCNN, dataset, summary, torch):
    model1 = SimpleCNN()
    # model1 = torch.compile(model1)
    model1(dataset[0][0]).shape
    summary(model1, input_size=(1, 232, 232))
    model1.opt = torch.optim.Adam(model1.parameters(), lr=0.001)
    return (model1,)


@app.cell
def _(DataLoader, dataset_test, dataset_valid):
    data_test = DataLoader(dataset_test, batch_size=64, num_workers=4, persistent_workers=True)
    data_valid = DataLoader(dataset_valid, batch_size=64, num_workers=4, persistent_workers=True)
    return data_test, data_valid


@app.cell
def _(DataLoader, Subset, dataset_train):
    data_loader = DataLoader(Subset(dataset_train, range(64)), batch_size=64, shuffle=True, num_workers=4, persistent_workers=True)
    return (data_loader,)


@app.cell
def _(mo, model1, model1_buttons):
    mo.stop(not model1_buttons["reset"].value)
    for module in model1.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
    return


@app.cell(hide_code=True)
def _(mo, model1, model1_buttons, my2, torch):
    # save
    mo.stop(not model1_buttons["save"].value)
    torch.save(model1.state_dict(), f"{my2.get_path(model1)}/1.pth")
    return


@app.cell(hide_code=True)
def _(mo, model1, model1_buttons, my2, torch):
    # load
    mo.stop(not model1_buttons["load"].value)
    model1.load_state_dict(torch.load(f"{my2.get_path(model1)}/1.pth", weights_only=True))
    return


@app.cell
def _(data_loader, mo, model1, model1_buttons, nn):
    # train
    mo.stop(not model1_buttons["train"].value)

    def _f():
        for epoc in range(10000):
            for batch, xy in enumerate(data_loader):  # note dont break in anyway or the worker will crash
                predict = model1(xy[0])
                loss = nn.functional.cross_entropy(predict, xy[1])
                loss.backward()
                model1.opt.step()
                model1.opt.zero_grad()
                print(loss)

    _f()
    return


@app.cell
def _(AUC, F, data_valid, mo, model1, model1_buttons, nn, torch):
    # valid
    mo.stop(not model1_buttons["valid"].value)

    @torch.no_grad()
    def _f():
        model1.train(False)
        loss = 0
        su = 0
        right = 0
        metric = AUC(n_tasks=3)
        for batch, xy in enumerate(data_valid):  # note dont break in anyway or the worker will crash
            predict = model1(xy[0])
            metric.update(predict.T, F.one_hot(xy[1], 3).T)
            loss += nn.functional.cross_entropy(predict, xy[1])
            right += (predict.argmax(-1) == xy[1]).sum()
            right += (predict.argmax(-1) != xy[1]).sum()
            su += predict.shape[0]
        print(loss)
        print(metric.compute())
        print(f"{right}/{su}")
        model1.train(True)

    _f()
    return


@app.cell
def _(AUC, F, data_test, mo, model1, model1_buttons, nn, torch):
    # test
    mo.stop(not model1_buttons["test"].value)

    @torch.no_grad()
    def _f():
        model1.train(False)
        loss = 0
        su = 0
        right = 0
        metric = AUC(n_tasks=3)
        wrong = []
        for batch, xy in enumerate(data_test):  # note dont break in anyway or the worker will crash
            predict = model1(xy[0])
            metric.update(predict.T, F.one_hot(xy[1], 3).T)
            loss += nn.functional.cross_entropy(predict, xy[1])
            right += (predict.argmax(-1) == xy[1]).sum()
            wrong.append((predict.argmax(-1) != xy[1]).nonzero().flatten() + batch * data_test.batch_size)
            su += predict.shape[0]
        wrong = torch.concat(wrong)
        print(wrong)
        print(loss)
        print(metric.compute())
        print(f"{right}/{su}")

        model1.train(True)

    _f()
    return


@app.cell
def _(dataset, dataset_test, transformi):
    print(dataset_test[0][1],dataset.classes)
    transformi(dataset_test[0][0])
    return


@app.cell
def _(model1_buttons):
    model1_buttons.hstack()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## TEST
    """)
    return


@app.cell
def _(model1, my2):
    my2.get_dir(model1)
    return


@app.cell
def _(nn):
    nn.Conv2d(1, 2, (3, 4), (5, 6), (7, 8), (9, 10))
    return


@app.cell
def _(nn):
    nn.MaxPool2d((1, 2), (3, 4), (5, 6), (7, 8))
    return


@app.cell
def _(dataset):
    dataset.__dir__()
    return


app._unparsable_cell(
    r"""
    mo.ui.range_slider(0, len(dataset_train),,full_width=True,label="train size")
    """,
    name="_"
)


if __name__ == "__main__":
    app.run()
