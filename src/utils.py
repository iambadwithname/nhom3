# Phần dùng chung cho các file 1_ -> 6_
# PyTorch không có sẵn model.fit() / model.evaluate() như Keras nên nhóm tự viết ở đây
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

CLASSES = ["COVID", "NORMAL", "PNEUMONIA"]
KAGGLE_ID = "sachinkumar413/covid-pneumonia-normal-chest-xray-images"
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS = ROOT / "results"
SPLIT_CSV = DATA_DIR / "splits.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def raw_dir():
    # dùng bản dataset đã tải về máy nếu có, chưa có thì tải từ Kaggle
    cached = sorted(Path.home().glob(f".cache/kagglehub/datasets/{KAGGLE_ID}/versions/*"), key=lambda p: int(p.name))
    if cached:
        return cached[-1]
    import kagglehub

    return Path(kagglehub.dataset_download(KAGGLE_ID))


def train_augment():
    # xoay, dịch, zoom, đổi sáng/tương phản; không lật ngang vì tim nằm bên trái
    return v2.Compose([
        v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        v2.ColorJitter(brightness=0.2, contrast=0.2),
    ])


class XrayDataset(Dataset):
    def __init__(self, images, labels, augment, rgb):
        self.images = images
        self.labels = torch.tensor(labels)
        self.aug = train_augment() if augment else None
        self.rgb = rgb
        # model pretrained (ResNet, MobileNet) thì chuẩn hóa theo ImageNet
        if rgb:
            self.norm = v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        else:
            self.norm = v2.Normalize([0.5], [0.25])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = self.images[i]
        if self.aug is not None:
            x = self.aug(x)
        x = x.float() / 255.0
        if self.rgb:
            x = x.expand(3, -1, -1)
        return self.norm(x), self.labels[i]


_cache = {}


def load_split(split, size):
    # đọc ảnh của 1 tập (train/val/test) theo data/splits.csv, chuyển xám, resize
    # lần đầu lưu ra data/cache_*.pt, các lần sau load lại cho nhanh
    if (split, size) in _cache:
        return _cache[split, size]
    cache_file = DATA_DIR / f"cache_{split}_{size}.pt"
    if cache_file.exists():
        _cache[split, size] = torch.load(cache_file)
        return _cache[split, size]
    src = raw_dir()
    with open(SPLIT_CSV) as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split]
    images = torch.empty(len(rows), 1, size, size, dtype=torch.uint8)
    for i, r in enumerate(rows):
        img = Image.open(src / r["file"]).convert("L").resize((size, size), Image.BILINEAR)
        images[i, 0] = torch.from_numpy(np.array(img))
    labels = [CLASSES.index(r["label"]) for r in rows]
    _cache[split, size] = (images, labels)
    torch.save(_cache[split, size], cache_file)
    return _cache[split, size]


def make_loader(split, augment, rgb, batch_size, size):
    images, labels = load_split(split, size)
    ds = XrayDataset(images, labels, augment=augment and split == "train", rgb=rgb)
    return DataLoader(ds, batch_size=batch_size, shuffle=split == "train")


def train_model(model, cfg, unfreeze=None):
    # giống model.fit() của Keras: train, theo dõi val loss, early stopping, lưu model tốt nhất ra best.pt
    # unfreeze: hàm mở khóa tầng cuối (chỉ dùng cho transfer learning, sau cfg["head_epochs"] epoch)
    out_dir = RESULTS / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(DEVICE)
    tr = make_loader("train", cfg["augment"], cfg["rgb"], cfg["batch_size"], cfg["img_size"])
    va = make_loader("val", False, cfg["rgb"], cfg["batch_size"], cfg["img_size"])

    # lớp nào ít ảnh (COVID) thì trọng số trong loss lớn hơn
    _, train_labels = load_split("train", cfg["img_size"])
    counts = np.bincount(train_labels, minlength=len(CLASSES))
    weights = torch.tensor(counts.sum() / (len(CLASSES) * counts), dtype=torch.float32, device=DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    scaler = torch.amp.GradScaler(enabled=DEVICE.type == "cuda")

    def new_optimizer(lr):
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
        return opt, torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.3, patience=3)

    optimizer, scheduler = new_optimizer(cfg["lr"])
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": [], "phase": []}
    best, bad_epochs, best_epoch = float("inf"), 0, 0
    t0 = time.time()
    for epoch in range(cfg["epochs"]):
        if unfreeze is None:
            phase = "full"
        elif epoch < cfg["head_epochs"]:
            phase = "head"
        else:
            phase = "finetune"
        # hết giai đoạn chỉ train lớp cuối thì mở khóa tầng cuối, fine-tune với lr nhỏ hơn
        if unfreeze is not None and epoch == cfg["head_epochs"]:
            unfreeze(model)
            optimizer, scheduler = new_optimizer(cfg["finetune_lr"])
            bad_epochs = 0

        result = []
        for loader, training in [(tr, True), (va, False)]:
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
            result += [total_loss / n, correct / n]
        tl, ta, vl, vacc = result
        scheduler.step(vl)
        for k, v in zip(history, [tl, ta, vl, vacc, optimizer.param_groups[0]["lr"], phase]):
            history[k].append(v)
        print(f"[{cfg['name']}] ep {epoch + 1:02d} {phase:8s} train {tl:.4f}/{ta:.3f}  val {vl:.4f}/{vacc:.3f}", flush=True)

        # early stopping theo val loss
        if vl < best - 1e-4:
            best, bad_epochs, best_epoch = vl, 0, epoch + 1
            torch.save(model.state_dict(), out_dir / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["patience"] and phase != "head":
                break

    meta = {
        "config": cfg,
        "best_epoch": best_epoch,
        "epochs_run": len(history["train_loss"]),
        "train_seconds": round(time.time() - t0, 1),
        "params_total": sum(p.numel() for p in model.parameters()),
        "params_trainable_final": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    (out_dir / "history.json").write_text(json.dumps(history, indent=1))
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    # nạp lại model tốt nhất để đánh giá
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=DEVICE))
    return model


@torch.no_grad()
def predict(model, split, rgb, size):
    # trả về nhãn thật + xác suất model đoán cho từng ảnh
    model = model.to(DEVICE).eval()
    ys, probs = [], []
    for x, y in make_loader(split, False, rgb, 64, size):
        probs.append(torch.softmax(model(x.to(DEVICE)).float(), 1).cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(probs).numpy()


def evaluate_model(model, name, split, rgb, size):
    # giống model.evaluate() của Keras: tính accuracy, precision, recall, F1, confusion matrix rồi lưu lại
    y, prob = predict(model, split, rgb, size)
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
    (RESULTS / name / f"metrics_{split}.json").write_text(json.dumps(res, indent=1))
    np.save(RESULTS / name / f"probs_{split}.npy", prob)
    print(f"[{name}] {split} acc {res['accuracy']:.4f} f1 {res['f1_macro']:.4f}")
    return res
