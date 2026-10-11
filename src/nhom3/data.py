import csv
import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

CLASSES = ["COVID", "NORMAL", "PNEUMONIA"]
KAGGLE_ID = "sachinkumar413/covid-pneumonia-normal-chest-xray-images"
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
SPLIT_CSV = DATA_DIR / "splits.csv"
IMG_SIZE = 224
SEED = 42


def raw_dir():
    # dùng bản đã tải về máy nếu có, khỏi tải lại
    cached = sorted(Path.home().glob(f".cache/kagglehub/datasets/{KAGGLE_ID}/versions/*"), key=lambda p: int(p.name))
    if cached:
        return cached[-1]
    import kagglehub

    return Path(kagglehub.dataset_download(KAGGLE_ID))


def dhash(path, size=16):
    # "dấu vân tay" của ảnh: so sánh độ sáng 2 pixel cạnh nhau
    img = Image.open(path).convert("L").resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(img, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def group_duplicates(hashes, threshold):
    # gom các ảnh có dHash khác nhau <= threshold bit vào cùng 1 nhóm
    parent = list(range(len(hashes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    dist = (hashes[:, None, :] != hashes[None, :, :]).sum(-1)
    for i, j in zip(*np.where(np.triu(dist <= threshold, 1))):
        parent[find(i)] = find(j)
    return [find(i) for i in range(len(hashes))]


def build_splits(threshold=4, ratios=(0.7, 0.15, 0.15)):
    src = raw_dir()
    files = sorted(p for c in CLASSES for p in (src / c).glob("*.png"))
    labels = [CLASSES.index(p.parent.name) for p in files]
    summary = {"raw": {c: labels.count(i) for i, c in enumerate(CLASSES)}}

    # 1. bỏ ảnh lỗi
    ok = []
    for p, y in zip(files, labels):
        try:
            Image.open(p).verify()
            ok.append((p, y))
        except Exception:
            pass
    summary["corrupt"] = len(files) - len(ok)

    # 2. bỏ ảnh trùng hoàn toàn
    seen, uniq = set(), []
    for p, y in ok:
        h = hashlib.md5(p.read_bytes()).hexdigest()
        if h not in seen:
            seen.add(h)
            uniq.append((p, y))
    summary["exact_dups"] = len(ok) - len(uniq)

    # 3. bỏ ảnh gần trùng, mỗi nhóm giữ 1 ảnh; nhóm nào lẫn nhãn thì bỏ cả nhóm
    roots = group_duplicates(np.stack([dhash(p) for p, _ in uniq]), threshold)
    groups = {}
    for (p, y), r in zip(uniq, roots):
        groups.setdefault(r, []).append((p, y))
    clean, conflicts = [], 0
    for members in groups.values():
        if len({y for _, y in members}) > 1:
            conflicts += len(members)
            continue
        clean.append(min(members, key=lambda m: m[0].name))
    summary["near_dups"] = len(uniq) - len(clean) - conflicts
    summary["label_conflicts"] = conflicts
    summary["clean"] = {c: sum(y == i for _, y in clean) for i, c in enumerate(CLASSES)}

    # 4. chia train/val/test theo từng lớp
    rng = np.random.default_rng(SEED)
    rows = []
    for i in range(len(CLASSES)):
        items = [p for p, y in clean if y == i]
        rng.shuffle(items)
        n_tr = round(len(items) * ratios[0])
        n_va = round(len(items) * ratios[1])
        for k, p in enumerate(items):
            split = "train" if k < n_tr else "val" if k < n_tr + n_va else "test"
            rows.append((f"{CLASSES[i]}/{p.name}", CLASSES[i], split))

    DATA_DIR.mkdir(exist_ok=True)
    with open(SPLIT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "label", "split"])
        w.writerows(sorted(rows))
    summary["split"] = {
        s: {c: sum(r[1] == c and r[2] == s for r in rows) for c in CLASSES}
        for s in ("train", "val", "test")
    }
    return summary


def read_split(split):
    src = raw_dir()
    with open(SPLIT_CSV) as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split]
    return [src / r["file"] for r in rows], [CLASSES.index(r["label"]) for r in rows]


def load_images(paths, size=IMG_SIZE):
    out = torch.empty(len(paths), 1, size, size, dtype=torch.uint8)
    for i, p in enumerate(paths):
        img = Image.open(p).convert("L").resize((size, size), Image.BILINEAR)
        out[i, 0] = torch.from_numpy(np.array(img))
    return out


def train_augment():
    # không lật ngang vì tim nằm bên trái
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
        # model pretrained thì chuẩn hóa theo ImageNet
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


def get_split(split, size=IMG_SIZE):
    # lần đầu đọc ảnh rồi lưu ra data/cache_*.pt, các lần sau load lại cho nhanh
    if (split, size) not in _cache:
        cache_file = DATA_DIR / f"cache_{split}_{size}.pt"
        if cache_file.exists():
            _cache[split, size] = torch.load(cache_file)
        else:
            paths, labels = read_split(split)
            _cache[split, size] = (load_images(paths, size), labels)
            torch.save(_cache[split, size], cache_file)
    return _cache[split, size]


def make_loader(split, augment, rgb, batch_size=32, size=IMG_SIZE):
    images, labels = get_split(split, size)
    ds = XrayDataset(images, labels, augment=augment and split == "train", rgb=rgb)
    return DataLoader(ds, batch_size=batch_size, shuffle=split == "train")
