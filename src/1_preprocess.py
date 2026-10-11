# Task 1: làm sạch dữ liệu và chia train / val / test
# Chạy: uv run python src/1_preprocess.py
import csv
import hashlib
import json

import numpy as np
from PIL import Image

from utils import CLASSES, DATA_DIR, SPLIT_CSV, raw_dir

# 1. Đọc danh sách ảnh
src = raw_dir()
files = sorted(p for c in CLASSES for p in (src / c).glob("*.png"))
labels = [CLASSES.index(p.parent.name) for p in files]
summary = {"raw": {c: labels.count(i) for i, c in enumerate(CLASSES)}}
print("Raw:", summary["raw"])

# 2. Bỏ ảnh lỗi (không mở được)
ok = []
for p, y in zip(files, labels):
    try:
        Image.open(p).verify()
        ok.append((p, y))
    except Exception:
        pass
summary["corrupt"] = len(files) - len(ok)

# 3. Bỏ ảnh trùng hoàn toàn (cùng mã MD5)
seen, uniq = set(), []
for p, y in ok:
    h = hashlib.md5(p.read_bytes()).hexdigest()
    if h not in seen:
        seen.add(h)
        uniq.append((p, y))
summary["exact_dups"] = len(ok) - len(uniq)

# 4. Bỏ ảnh gần trùng
# mỗi ảnh -> "dấu vân tay" 16x16 (dHash): so độ sáng 2 pixel cạnh nhau
hashes = []
for p, _ in uniq:
    a = np.asarray(Image.open(p).convert("L").resize((17, 16), Image.BILINEAR), dtype=np.int16)
    hashes.append((a[:, 1:] > a[:, :-1]).flatten())
hashes = np.stack(hashes)

# 2 ảnh có dấu vân tay khác nhau <= 4 bit thì coi là cùng 1 ảnh, gom thành nhóm
parent = list(range(len(hashes)))


def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


dist = (hashes[:, None, :] != hashes[None, :, :]).sum(-1)
for i, j in zip(*np.where(np.triu(dist <= 4, 1))):
    parent[find(i)] = find(j)

groups = {}
for k, (p, y) in enumerate(uniq):
    groups.setdefault(find(k), []).append((p, y))

# mỗi nhóm giữ 1 ảnh; nhóm nào lẫn nhãn khác nhau thì bỏ cả nhóm
clean, conflicts = [], 0
for members in groups.values():
    if len({y for _, y in members}) > 1:
        conflicts += len(members)
        continue
    clean.append(min(members, key=lambda m: m[0].name))
summary["near_dups"] = len(uniq) - len(clean) - conflicts
summary["label_conflicts"] = conflicts
summary["clean"] = {c: sum(y == i for _, y in clean) for i, c in enumerate(CLASSES)}

# 5. Chia train / val / test = 70 / 15 / 15, chia riêng từng lớp, seed 42
rng = np.random.default_rng(42)
rows = []
for i in range(len(CLASSES)):
    items = [p for p, y in clean if y == i]
    rng.shuffle(items)
    n_tr = round(len(items) * 0.7)
    n_va = round(len(items) * 0.15)
    for k, p in enumerate(items):
        split = "train" if k < n_tr else "val" if k < n_tr + n_va else "test"
        rows.append((f"{CLASSES[i]}/{p.name}", CLASSES[i], split))

# 6. Lưu kết quả
DATA_DIR.mkdir(exist_ok=True)
with open(SPLIT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["file", "label", "split"])
    w.writerows(sorted(rows))
summary["split"] = {
    s: {c: sum(r[1] == c and r[2] == s for r in rows) for c in CLASSES}
    for s in ("train", "val", "test")
}
(DATA_DIR / "cleaning_summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
