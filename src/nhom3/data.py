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


def raw_dir() -> Path:
    import kagglehub

    return Path(kagglehub.dataset_download(KAGGLE_ID))


def dhash(path: Path, size: int = 16) -> np.ndarray:
    img = Image.open(path).convert("L").resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(img, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def _groups(hashes: np.ndarray, threshold: int) -> list[int]:
    """Union-find over images whose dHash Hamming distance <= threshold."""
    parent = list(range(len(hashes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    dist = (hashes[:, None, :] != hashes[None, :, :]).sum(-1)
    for i, j in zip(*np.where(np.triu(dist <= threshold, 1))):
        parent[find(i)] = find(j)
    return [find(i) for i in range(len(hashes))]


def build_splits(threshold: int = 4, ratios=(0.7, 0.15, 0.15)) -> dict:
    """Clean the raw dataset and write data/splits.csv. Returns a summary dict."""
    src = raw_dir()
    files = sorted(p for c in CLASSES for p in (src / c).glob("*.png"))
    labels = [CLASSES.index(p.parent.name) for p in files]
    summary = {"raw": {c: labels.count(i) for i, c in enumerate(CLASSES)}}

    # 1) unreadable files
    ok = []
    for p, y in zip(files, labels):
        try:
            Image.open(p).verify()
            ok.append((p, y))
        except Exception:
            pass
    summary["corrupt"] = len(files) - len(ok)

    # 2) exact duplicates (same bytes)
    seen, uniq = set(), []
    for p, y in ok:
        h = hashlib.md5(p.read_bytes()).hexdigest()
        if h not in seen:
            seen.add(h)
            uniq.append((p, y))
    summary["exact_dups"] = len(ok) - len(uniq)

    # 3) near duplicates (resized / lightly edited copies). Keep one image per group,
    #    drop the whole group if it mixes labels.
    roots = _groups(np.stack([dhash(p) for p, _ in uniq]), threshold)
    by_group: dict[int, list] = {}
    for (p, y), r in zip(uniq, roots):
        by_group.setdefault(r, []).append((p, y))
    clean, conflicts = [], 0
    for members in by_group.values():
        if len({y for _, y in members}) > 1:
            conflicts += len(members)
            continue
        clean.append(min(members, key=lambda m: m[0].name))
    summary["near_dups"] = len(uniq) - len(clean) - conflicts
    summary["label_conflicts"] = conflicts
    summary["clean"] = {c: sum(y == i for _, y in clean) for i, c in enumerate(CLASSES)}

    # 4) stratified split
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


def read_split(split: str) -> tuple[list[Path], list[int]]:
    src = raw_dir()
    with open(SPLIT_CSV) as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split]
    return [src / r["file"] for r in rows], [CLASSES.index(r["label"]) for r in rows]


def load_images(paths: list[Path]) -> torch.Tensor:
    # uint8 (N, 1, H, W), whole dataset is only ~250 MB so keep it in RAM
    out = torch.empty(len(paths), 1, IMG_SIZE, IMG_SIZE, dtype=torch.uint8)
    for i, p in enumerate(paths):
        img = Image.open(p).convert("L").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        out[i, 0] = torch.from_numpy(np.array(img))
    return out


def train_augment() -> v2.Compose:
    # no horizontal flip (heart is on the left side)
    return v2.Compose(
        [
            v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            v2.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    )


class XrayDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: list[int], augment: bool, rgb: bool):
        self.images = images
        self.labels = torch.tensor(labels)
        self.aug = train_augment() if augment else None
        # rgb=True -> ImageNet statistics for pretrained backbones
        mean, std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) if rgb else ([0.5], [0.25])
        self.rgb = rgb
        self.norm = v2.Normalize(mean, std)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        x = self.images[i]
        if self.aug is not None:
            x = self.aug(x)
        x = x.float() / 255.0
        if self.rgb:
            x = x.expand(3, -1, -1)
        return self.norm(x), self.labels[i]


_cache: dict[str, tuple[torch.Tensor, list[int]]] = {}


def get_split(split: str) -> tuple[torch.Tensor, list[int]]:
    if split not in _cache:
        cache_file = DATA_DIR / f"cache_{split}_{IMG_SIZE}.pt"
        if cache_file.exists():
            _cache[split] = torch.load(cache_file)
        else:
            paths, labels = read_split(split)
            _cache[split] = (load_images(paths), labels)
            torch.save(_cache[split], cache_file)
    return _cache[split]


def make_loader(split: str, augment: bool, rgb: bool, batch_size: int = 32) -> DataLoader:
    images, labels = get_split(split)
    ds = XrayDataset(images, labels, augment=augment and split == "train", rgb=rgb)
    return DataLoader(ds, batch_size=batch_size, shuffle=split == "train", num_workers=0)
