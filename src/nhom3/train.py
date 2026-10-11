import json
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from torch import nn

from . import models
from .data import CLASSES, ROOT, get_split, make_loader

RESULTS = ROOT / "results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Config:
    name: str
    arch: str  # simple / complex / resnet50 / mobilenet
    augment: bool = True
    lr: float = 1e-3
    batch_size: int = 32
    epochs: int = 40
    patience: int = 8
    weight_decay: float = 1e-4
    head_epochs: int = 5  # transfer learning: số epoch chỉ train lớp cuối
    finetune_lr: float = 1e-4
    seed: int = 42
    img_size: int = 224

    @property
    def rgb(self):
        return self.arch in ("resnet50", "mobilenet")


def build_model(cfg):
    if cfg.arch == "simple":
        return models.SimpleCNN(img_size=cfg.img_size)
    if cfg.arch == "complex":
        return models.ComplexCNN()
    if cfg.arch == "resnet50":
        return models.resnet50_transfer()
    return models.mobilenet_transfer()


def class_weights():
    # lớp nào ít ảnh (COVID) thì trọng số lớn hơn
    _, labels = get_split("train")
    counts = np.bincount(labels, minlength=len(CLASSES))
    w = counts.sum() / (len(CLASSES) * counts)
    return torch.tensor(w, dtype=torch.float32, device=DEVICE)


def run_epoch(model, loader, loss_fn, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
                out = model(x)
                loss = loss_fn(out, y)
            if training:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


def make_optimizer(model, lr, wd):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.3, patience=3)
    return opt, sched


def train(cfg):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    out_dir = RESULTS / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg).to(DEVICE)
    tr = make_loader("train", cfg.augment, cfg.rgb, cfg.batch_size, cfg.img_size)
    va = make_loader("val", False, cfg.rgb, cfg.batch_size, cfg.img_size)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights())
    scaler = torch.amp.GradScaler(enabled=DEVICE.type == "cuda")
    optimizer, scheduler = make_optimizer(model, cfg.lr, cfg.weight_decay)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": [], "phase": []}
    best, bad_epochs, best_epoch = float("inf"), 0, 0
    t0 = time.time()
    for epoch in range(cfg.epochs):
        if not cfg.rgb:
            phase = "full"
        elif epoch < cfg.head_epochs:
            phase = "head"
        else:
            phase = "finetune"
        # hết giai đoạn train lớp cuối thì mở khóa và fine-tune với lr nhỏ hơn
        if cfg.rgb and epoch == cfg.head_epochs:
            models.unfreeze_last_stages(model)
            optimizer, scheduler = make_optimizer(model, cfg.finetune_lr, cfg.weight_decay)
            bad_epochs = 0

        tl, ta = run_epoch(model, tr, loss_fn, optimizer, scaler)
        vl, vacc = run_epoch(model, va, loss_fn)
        scheduler.step(vl)
        for k, v in zip(history, [tl, ta, vl, vacc, optimizer.param_groups[0]["lr"], phase]):
            history[k].append(v)
        print(f"[{cfg.name}] ep {epoch + 1:02d} {phase:8s} train {tl:.4f}/{ta:.3f}  val {vl:.4f}/{vacc:.3f}", flush=True)

        # early stopping theo val loss, lưu lại model tốt nhất
        if vl < best - 1e-4:
            best, bad_epochs, best_epoch = vl, 0, epoch + 1
            torch.save(model.state_dict(), out_dir / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience and phase != "head":
                break

    total, trainable = models.count_params(model)
    meta = {
        "config": asdict(cfg),
        "best_epoch": best_epoch,
        "epochs_run": len(history["train_loss"]),
        "train_seconds": round(time.time() - t0, 1),
        "params_total": total,
        "params_trainable_final": trainable,
    }
    (out_dir / "history.json").write_text(json.dumps(history, indent=1))
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def load_trained(cfg):
    model = build_model(cfg)
    if cfg.rgb:
        models.unfreeze_last_stages(model)
    model.load_state_dict(torch.load(RESULTS / cfg.name / "best.pt", map_location=DEVICE))
    return model.to(DEVICE).eval()


@torch.no_grad()
def predict(model, split, rgb, size=224):
    ys, probs = [], []
    for x, y in make_loader(split, False, rgb, 64, size):
        probs.append(torch.softmax(model(x.to(DEVICE)).float(), 1).cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(probs).numpy()


def evaluate(cfg, split="test"):
    model = load_trained(cfg)
    y, prob = predict(model, split, cfg.rgb, cfg.img_size)
    pred = prob.argmax(1)
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    res = {
        "accuracy": accuracy_score(y, pred),
        "precision_macro": p,
        "recall_macro": r,
        "f1_macro": f,
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "report": classification_report(y, pred, target_names=CLASSES, output_dict=True, zero_division=0),
    }
    out_dir = RESULTS / cfg.name
    (out_dir / f"metrics_{split}.json").write_text(json.dumps(res, indent=1))
    np.save(out_dir / f"probs_{split}.npy", prob)
    return res


def run(cfg):
    meta = train(cfg)
    res = evaluate(cfg, "test")
    evaluate(cfg, "val")
    print(f"[{cfg.name}] test acc {res['accuracy']:.4f} f1 {res['f1_macro']:.4f}  ({meta['train_seconds']}s)")
    return res


EXPERIMENTS = [
    Config("simple_cnn", "simple", augment=False, lr=1e-3, batch_size=64, img_size=128),
    Config("simple_cnn_aug", "simple", augment=True, lr=1e-3, batch_size=64, img_size=128),
    Config("complex_cnn_aug", "complex", augment=True),
    Config("resnet50_tl", "resnet50", augment=True, epochs=30),
    Config("mobilenetv3_tl", "mobilenet", augment=True, epochs=30),
]

# dò lr và batch size cho Simple CNN, chọn theo F1 trên tập val
TUNING = [
    Config(f"tune_simple_lr{lr:g}_bs{bs}", "simple", lr=lr, batch_size=bs, epochs=25, patience=6, img_size=128)
    for lr in (3e-3, 1e-3, 3e-4)
    for bs in (32, 64)
]


def by_name(name):
    return next(c for c in EXPERIMENTS + TUNING if c.name == name)
