import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full", auto_download=["html"])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup
    """)
    return


@app.cell(hide_code=True)
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
    from torchmetrics import classification
    import datetime
    import time
    import hvplot.polars

    return (
        DataLoader,
        Subset,
        classification,
        dataset,
        datetime,
        islice,
        mo,
        my2,
        nn,
        pl,
        random_split,
        summary,
        time,
        torch,
        torchvision,
        transformi,
    )


@app.cell
def _(torch):
    # setup
    # torch._logging.set_logs(graph_code=False)
    pin_memory = True
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        pin_memory = False
    batch_size = 64
    data_loader_param = dict(batch_size=batch_size, persistent_workers=True, pin_memory=pin_memory)
    return data_loader_param, device


@app.cell
def _(torchvision):
    # for data aumentation
    torchvision.transforms.__dir__()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## DATA
    """)
    return


@app.cell(hide_code=True)
def _(dataset):
    dataset[0][0].shape
    return


@app.cell(hide_code=True)
def _(dataset, random_split, torch):
    def _f_split(dataset):
        n = len(dataset)
        l = [int(n * 0.7), int(n * 0.1)]  # why 622 cause duh
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
            self.drop = nn.Dropout(0.09)
            self.Activation = nn.Sequential(nn.ReLU())
            self.CNN = nn.Sequential(
                nn.Conv2d(1, 2, (3, 3)),
                nn.Conv2d(2, 4, (3, 3), (1, 1), (3, 3)),
                self.drop,
                nn.ReLU(),
                nn.Conv2d(4, 8, (3, 3), (3, 3)),
                self.drop,
                nn.ReLU(),
                nn.Conv2d(8, 16, (3, 3), (3, 3)),
                nn.ReLU(),
                self.drop,
                nn.Conv2d(16, 32, (3, 3), (3, 3)),
                self.Activation,
                nn.MaxPool2d((3, 3), (1, 1), (1, 1)),
                nn.Conv2d(32, 64, (3, 3), (3, 3)),
            )
            self.pooling = nn.AdaptiveAvgPool2d((2, 2))
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
            return x

    model1_buttons = mo.ui.dictionary({"train size": mo.ui.range_slider(0, len(dataset_train), value=(0, 256), full_width=True, label="train size")} | {i: mo.ui.run_button(label=i) for i in ["train", "test", "valid", "load", "save", "reset"]})
    return SimpleCNN, model1_buttons


@app.cell(hide_code=True)
def _(SimpleCNN, summary):
    model1 = SimpleCNN()
    # model1 = torch.compile(model1)
    _ = summary(model1, input_size=(1, 232, 232))
    return (model1,)


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(DataLoader, data_loader_param, dataset_test, dataset_valid):
    data_test = DataLoader(dataset_test, num_workers=1, **data_loader_param)
    data_valid = DataLoader(dataset_valid, num_workers=1, **data_loader_param)
    return data_test, data_valid


@app.cell(hide_code=True)
def _(DataLoader, Subset, data_loader_param, dataset_train, model1_buttons):
    data_loader = DataLoader(Subset(dataset_train, range(*model1_buttons["train size"].value)), shuffle=True, num_workers=4, **data_loader_param)
    return (data_loader,)


@app.cell(hide_code=True)
def _(mo, model1, model1_buttons):
    # reset
    mo.stop(not model1_buttons["reset"].value)
    for module in model1.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
    return


@app.cell(hide_code=True)
def _(mo, model1, model1_buttons, my2, torch):
    # save
    mo.stop(not model1_buttons["save"].value)
    torch.save(model1.state_dict(), f"{my2.get_dir(model1)}/1.pth")
    print(f"{my2.get_dir(model1)}/1.pth")
    return


@app.cell(hide_code=True)
def _(model1, model1_buttons, my2, torch):
    # load
    model1_buttons["load"].value
    model1.load_state_dict(torch.load(f"{my2.get_dir(model1)}/1.pth", weights_only=True))
    return


@app.cell(hide_code=True)
def _(classification, data_valid, device, nn, torch):
    @torch.no_grad()
    def valid(model):
        model.train(False)
        loss = 0
        su = 0
        right = 0
        auc = classification.MulticlassAUROC(3, average=None)
        recall = classification.MulticlassRecall(3, average=None).to(device)
        for batch, xy in enumerate(data_valid):  # note dont break in anyway or the worker will crash
            xy = [i.to(device) for i in xy]
            predict = model(xy[0])
            auc.update(predict, xy[1])
            recall.update(predict, xy[1])
            loss += nn.CrossEntropyLoss()(predict, xy[1])
            right += (predict.argmax(-1) == xy[1]).sum()
            su += predict.shape[0]
        print(loss)
        print(auc.compute())
        print("recall",recall.compute())
        print(f"{right}/{su}")
        model.train(True)
        return (loss, auc.compute(), right / su)

    return (valid,)


@app.cell
def _(mo, model1, model1_buttons, valid):
    # valid
    mo.stop(not model1_buttons["valid"].value)
    valid(model1)
    return


@app.cell
def _(pl):
    pl.DataFrame()
    return


@app.cell
def _():
    return


@app.cell
def _(
    data_loader,
    datetime,
    device,
    mo,
    model1,
    model1_buttons,
    my2,
    nn,
    pl,
    time,
    valid,
):
    # train
    mo.stop(not model1_buttons["train"].value)

    def _f():
        epoc = range(10)
        train_stat = pl.DataFrame({"epoc": epoc})
        train_stat = train_stat.with_columns(time_train=pl.lit(None, pl.Duration("ns")), train_loss=pl.lit(None, pl.Float64), valid_loss=pl.lit(None, pl.Float64), accuracy=pl.lit(None, pl.Float64), **{f"auc_{i}":pl.lit(None, pl.Float64) for i in range(3)})
        for epoc in epoc:
            loss2 = 0
            start = time.time()
            for batch, xy in enumerate(data_loader):  # note dont break in anyway or the worker will crash
                xy = [i.to(device) for i in xy]
                predict = model1(xy[0])
                loss = nn.CrossEntropyLoss()(predict, xy[1])
                loss.backward()
                model1.opt.step()
                model1.opt.zero_grad()
                loss2 += loss.detach()
            train_stat[epoc,"train_loss"] = loss2.to("cpu")
            train_stat[epoc,"time_train"] = time.time() - start
            print(loss2)
            if epoc % 5 == 4:
                (loss, auc, accu) = valid(model1)
                train_stat[epoc,"valid_loss"] = loss.to("cpu")
                for i in range(3):
                    train_stat[epoc,f"auc_{i}"] = auc.to("cpu")[i]
                train_stat[epoc,"accuracy"] = accu.to("cpu")
        train_stat.write_parquet(f"{my2.get_dir(model1)}/{str(datetime.datetime.now())}")
        return train_stat

    train_stat = _f()
    return (train_stat,)


@app.cell
def _(train_stat):
    train_stat[4::5]
    return


@app.cell
def _(train_stat):
    train_stat[4::5].hvplot.line(x="epoc",y=[*[f"auc_{i}"for i in range(3)]])
    return


@app.cell
def _(model1, torch):
    model1.opt = torch.optim.Adam(model1.parameters(), lr=0.0002)
    return


@app.cell(hide_code=True)
def _(
    classification,
    data_test,
    device,
    mo,
    model1,
    model1_buttons,
    nn,
    torch,
):
    # test
    mo.stop(not model1_buttons["test"].value)

    @torch.no_grad()
    def _f():
        model1.train(False)
        loss = 0
        su = 0
        right = 0
        auc = classification.MulticlassAUROC(3, average=None)
        recall = classification.MulticlassRecall(3, average=None).to(device)
        wrong = []
        for batch, xy in enumerate(data_test):  # note dont break in anyway or the worker will crash
            xy = [i.to(device) for i in xy]
            predict = model1(xy[0])
            auc.update(predict, xy[1])
            recall.update(predict, xy[1])
            loss += nn.functional.cross_entropy(predict, xy[1])
            right += (predict.argmax(-1) == xy[1]).sum()
            wrong.append((predict.argmax(-1) != xy[1]).nonzero().flatten() + batch * data_test.batch_size)
            su += predict.shape[0]
        wrong = torch.concat(wrong)
        # print(wrong)
        print(loss)
        print("auc",auc.compute())
        print("recall",recall.compute())
        print(f"{right}/{su}")
        print(f"{right / su}")
        print(loss)
        model1.train(True)

    _f()
    return


@app.cell(hide_code=True)
def _(model1_buttons):
    model1_buttons.hstack()
    return


@app.cell
def _(dataset, dataset_test, transformi):
    print(dataset_test[0][1], dataset.classes)
    transformi(dataset_test[0][0])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## TEST
    """)
    return


@app.cell
def _(metrics, torch):
    def _f():
        metric = metrics.aggregation.auc.AUC()
        # metric.update(torch.tensor([1]),torch.tensor([1]))
        # metric.update(torch.tensor([2]),torch.tensor([2]))
        # metric.update(torch.tensor([3,2,1]),torch.tensor([1,2,3]))
        metric.update(torch.tensor([1, 2, 3, 40]), torch.tensor([1, 2, 3, 4]))
        print(metric.compute())
        return metric.state_dict()

    _f()
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


if __name__ == "__main__":
    app.run()
