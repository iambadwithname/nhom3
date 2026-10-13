import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import csv
    import hashlib
    import json
    import sys

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from matplotlib.patches import FancyBboxPatch
    from PIL import Image
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

    sys.path.insert(0, str(mo.notebook_dir()))
    from models import ComplexCNN, SimpleCNN, mobilenet_transfer, unfreeze_last_stages
    from utils import (CLASSES, DATA_DIR, DEVICE, RESULTS, SPLIT_CSV, XrayDataset, evaluate_model, load_split,
                       predict, raw_dir, train_augment, train_model)

    FIG = RESULTS / "figures"
    return (
        CLASSES,
        ComplexCNN,
        DATA_DIR,
        DEVICE,
        FIG,
        FancyBboxPatch,
        Image,
        RESULTS,
        SPLIT_CSV,
        SimpleCNN,
        XrayDataset,
        accuracy_score,
        confusion_matrix,
        csv,
        evaluate_model,
        hashlib,
        json,
        load_split,
        mo,
        mobilenet_transfer,
        np,
        plt,
        precision_recall_fscore_support,
        predict,
        raw_dir,
        torch,
        train_augment,
        train_model,
        unfreeze_last_stages,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Chest X-ray Classification

    COVID-19 / Normal / Pneumonia. Các bước train chỉ chạy khi bấm nút; không bấm thì dùng kết quả đã lưu trong `results/`.

    ## 1. Tiền xử lý dữ liệu
    """)
    return


@app.cell
def _(mo):
    btn_prep = mo.ui.run_button(label="Chạy tiền xử lý")
    btn_prep
    return (btn_prep,)


@app.cell
def _(CLASSES, Image, btn_prep, hashlib, mo, raw_dir):
    mo.stop(not btn_prep.value)

    # 1. Đọc danh sách ảnh
    src = raw_dir()
    files = sorted(p for c in CLASSES for p in (src / c).glob("*.png"))
    labels = [CLASSES.index(p.parent.name) for p in files]

    # 2. Bỏ ảnh lỗi
    ok = []
    for _p, _y in zip(files, labels):
        try:
            Image.open(_p).verify()
            ok.append((_p, _y))
        except Exception:
            pass

    # 3. Bỏ ảnh trùng hoàn toàn (cùng mã MD5)
    _seen, uniq = set(), []
    for _p, _y in ok:
        _h = hashlib.md5(_p.read_bytes()).hexdigest()
        if _h not in _seen:
            _seen.add(_h)
            uniq.append((_p, _y))
    return files, labels, ok, uniq


@app.cell
def _(Image, np, uniq):
    # 4. Bỏ ảnh gần trùng: dấu vân tay 16x16 (dHash) khác nhau <= 4 bit thì coi là 1 ảnh
    _hashes = []
    for _p, _ in uniq:
        _a = np.asarray(Image.open(_p).convert("L").resize((17, 16), Image.BILINEAR), dtype=np.int16)
        _hashes.append((_a[:, 1:] > _a[:, :-1]).flatten())
    _hashes = np.stack(_hashes)

    _parent = list(range(len(_hashes)))

    def _find(i):
        while _parent[i] != i:
            _parent[i] = _parent[_parent[i]]
            i = _parent[i]
        return i

    _dist = (_hashes[:, None, :] != _hashes[None, :, :]).sum(-1)
    for _i, _j in zip(*np.where(np.triu(_dist <= 4, 1))):
        _parent[_find(_i)] = _find(_j)
    _groups = {}
    for _k, (_p, _y) in enumerate(uniq):
        _groups.setdefault(_find(_k), []).append((_p, _y))

    # mỗi nhóm giữ 1 ảnh; nhóm nào lẫn nhãn thì bỏ cả nhóm
    clean, conflicts = [], 0
    for _members in _groups.values():
        if len({_y for _, _y in _members}) > 1:
            conflicts += len(_members)
        else:
            clean.append(min(_members, key=lambda m: m[0].name))
    return clean, conflicts


@app.cell
def _(CLASSES, DATA_DIR, SPLIT_CSV, clean, conflicts, csv, files, json, labels, np, ok, uniq):
    # 5. Chia train / val / test = 70 / 15 / 15 theo từng lớp, seed 42
    _rng = np.random.default_rng(42)
    _rows = []
    for _i, _c in enumerate(CLASSES):
        _items = [p for p, y in clean if y == _i]
        _rng.shuffle(_items)
        _n_tr, _n_va = round(len(_items) * 0.7), round(len(_items) * 0.15)
        for _k, _p in enumerate(_items):
            _split = "train" if _k < _n_tr else "val" if _k < _n_tr + _n_va else "test"
            _rows.append((f"{_c}/{_p.name}", _c, _split))

    # 6. Lưu kết quả
    with open(SPLIT_CSV, "w", newline="") as _f:
        _w = csv.writer(_f)
        _w.writerow(["file", "label", "split"])
        _w.writerows(sorted(_rows))
    _summary = {
        "raw": {c: labels.count(i) for i, c in enumerate(CLASSES)},
        "corrupt": len(files) - len(ok),
        "exact_dups": len(ok) - len(uniq),
        "near_dups": len(uniq) - len(clean) - conflicts,
        "label_conflicts": conflicts,
        "clean": {c: sum(y == i for _, y in clean) for i, c in enumerate(CLASSES)},
        "split": {s: {c: sum(r[1] == c and r[2] == s for r in _rows) for c in CLASSES} for s in ("train", "val", "test")},
    }
    (DATA_DIR / "cleaning_summary.json").write_text(json.dumps(_summary, indent=1))
    return


@app.cell
def _(DATA_DIR, json, mo):
    _s = json.loads((DATA_DIR / "cleaning_summary.json").read_text())
    mo.ui.table([
        {"Bước": "Ảnh gốc", "Số ảnh": sum(_s["raw"].values())},
        {"Bước": "Ảnh lỗi", "Số ảnh": _s["corrupt"]},
        {"Bước": "Trùng hoàn toàn", "Số ảnh": _s["exact_dups"]},
        {"Bước": "Gần trùng", "Số ảnh": _s["near_dups"]},
        {"Bước": "Trùng khác nhãn", "Số ảnh": _s["label_conflicts"]},
        {"Bước": "Còn lại", "Số ảnh": sum(_s["clean"].values())},
    ], selection=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Simple CNN
    Conv 32 → Conv 32 → MaxPool → Flatten → Dense 128 → Dropout 0.5 → Dense 3, ảnh 128×128.
    """)
    return


@app.cell
def _(mo):
    btn_simple = mo.ui.run_button(label="Train Simple CNN")
    btn_simple
    return (btn_simple,)


@app.cell
def _(SimpleCNN, btn_simple, evaluate_model, mo, torch, train_model):
    mo.stop(not btn_simple.value)

    # 1. Dò learning rate và batch size trên tập val
    for _lr in (3e-3, 1e-3, 3e-4):
        for _bs in (32, 64):
            _cfg = dict(name=f"tune_simple_lr{_lr:g}_bs{_bs}", augment=True, lr=_lr, batch_size=_bs,
                        epochs=25, patience=6, img_size=128, rgb=False)
            torch.manual_seed(42)
            _model = train_model(SimpleCNN(img_size=128), _cfg)
            evaluate_model(_model, _cfg["name"], "val", rgb=False, size=128)

    # 2. Train với lr 0.001, batch 64: không aug và có aug, 3. đánh giá
    for _name, _augment in [("simple_cnn", False), ("simple_cnn_aug", True)]:
        _cfg = dict(name=_name, augment=_augment, lr=1e-3, batch_size=64,
                    epochs=40, patience=8, img_size=128, rgb=False)
        torch.manual_seed(42)
        _model = train_model(SimpleCNN(img_size=128), _cfg)
        evaluate_model(_model, _name, "test", rgb=False, size=128)
        evaluate_model(_model, _name, "val", rgb=False, size=128)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Complex CNN
    Stem → 2 khối Inception → 4 khối Residual → Global AvgPool → Dense, ảnh 224×224.
    """)
    return


@app.cell
def _(mo):
    btn_complex = mo.ui.run_button(label="Train Complex CNN")
    btn_complex
    return (btn_complex,)


@app.cell
def _(ComplexCNN, btn_complex, evaluate_model, mo, torch, train_model):
    mo.stop(not btn_complex.value)

    _cfg = dict(name="complex_cnn_aug", augment=True, lr=1e-3, batch_size=32,
                epochs=40, patience=8, img_size=224, rgb=False)
    torch.manual_seed(42)
    _model = train_model(ComplexCNN(), _cfg)
    evaluate_model(_model, _cfg["name"], "test", rgb=False, size=224)
    evaluate_model(_model, _cfg["name"], "val", rgb=False, size=224)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Transfer learning – MobileNetV3
    5 epoch đầu chỉ train lớp cuối (lr 0.001), sau đó fine-tune 4 khối cuối (lr 0.0001).
    """)
    return


@app.cell
def _(mo):
    btn_tl = mo.ui.run_button(label="Train MobileNetV3")
    btn_tl
    return (btn_tl,)


@app.cell
def _(btn_tl, evaluate_model, mo, mobilenet_transfer, torch, train_model, unfreeze_last_stages):
    mo.stop(not btn_tl.value)

    _cfg = dict(name="mobilenetv3_tl", augment=True, lr=1e-3, batch_size=32, epochs=30, patience=8,
                img_size=224, rgb=True, head_epochs=5, finetune_lr=1e-4)
    torch.manual_seed(42)
    _model = train_model(mobilenet_transfer(), _cfg, unfreeze=unfreeze_last_stages)
    evaluate_model(_model, _cfg["name"], "test", rgb=True, size=224)
    evaluate_model(_model, _cfg["name"], "val", rgb=True, size=224)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Dự đoán bằng model đã lưu
    Nạp `best.pt` của từng model rồi dự đoán tập test (627 ảnh), không train lại.
    """)
    return


@app.cell
def _(ComplexCNN, DEVICE, RESULTS, SimpleCNN, mobilenet_transfer, torch):
    models = []
    for _name, _model, _size, _rgb in [
        ("simple_cnn", SimpleCNN(img_size=128), 128, False),
        ("simple_cnn_aug", SimpleCNN(img_size=128), 128, False),
        ("complex_cnn_aug", ComplexCNN(), 224, False),
        ("mobilenetv3_tl", mobilenet_transfer(), 224, True),
    ]:
        _model.load_state_dict(torch.load(RESULTS / _name / "best.pt", map_location=DEVICE))
        models.append((_name, _model.to(DEVICE).eval(), _size, _rgb))
    return (models,)


@app.cell
def _(CLASSES, accuracy_score, confusion_matrix, mo, models, precision_recall_fscore_support, predict):
    _scores, _matrices = [], []
    for _name, _model, _size, _rgb in models:
        _y, _prob = predict(_model, "test", _rgb, _size)
        _pred = _prob.argmax(1)
        _p, _r, _f, _ = precision_recall_fscore_support(_y, _pred, average="macro", zero_division=0)
        _scores.append({"Model": _name, "Accuracy": round(accuracy_score(_y, _pred), 4),
                        "Precision": round(_p, 4), "Recall": round(_r, 4), "F1": round(_f, 4)})
        _cm = confusion_matrix(_y, _pred)
        _matrices.append(mo.vstack([
            mo.md(f"**{_name}**"),
            mo.ui.table([{"Thật \\ Đoán": c, **dict(zip(CLASSES, map(int, row)))} for c, row in zip(CLASSES, _cm)],
                        selection=None),
        ]))
    mo.vstack([mo.ui.table(_scores, selection=None), mo.hstack(_matrices, wrap=True)])
    return


@app.cell
def _(mo):
    img_path = mo.ui.text(label="Dự đoán 1 ảnh (đường dẫn file):", full_width=True)
    img_path
    return (img_path,)


@app.cell
def _(CLASSES, DEVICE, Image, XrayDataset, img_path, mo, models, np, torch):
    mo.stop(not img_path.value)

    _file = img_path.value.strip('"')
    _out = []
    for _name, _model, _size, _rgb in models:
        _img = Image.open(_file).convert("L").resize((_size, _size), Image.BILINEAR)
        _x = XrayDataset(torch.from_numpy(np.array(_img))[None, None], [0], augment=False, rgb=_rgb)[0][0]
        with torch.no_grad():
            _p = torch.softmax(_model(_x[None].to(DEVICE)).float(), 1)[0].cpu().numpy()
        _out.append({"Model": _name, "Đoán": CLASSES[_p.argmax()], "Xác suất": f"{_p.max():.2%}"})
    mo.hstack([mo.image(_file, width=220), mo.ui.table(_out, selection=None)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Hình cho báo cáo
    """)
    return


@app.cell
def _(FancyBboxPatch, plt):
    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3, "savefig.dpi": 200, "savefig.bbox": "tight"})
    BLUE, ORANGE, INK, MUTED = "tab:blue", "tab:orange", "black", "dimgray"
    CONV, POOL, FC, IO, BLOCK, FROZEN = "lightblue", "peachpuff", "palegreen", "whitesmoke", "lavender", "lightgray"
    NAMES = ["simple_cnn", "simple_cnn_aug", "complex_cnn_aug", "mobilenetv3_tl"]
    LABELS = {"simple_cnn": "Simple CNN", "simple_cnn_aug": "Simple CNN + Aug",
              "complex_cnn_aug": "Complex CNN + Aug", "mobilenetv3_tl": "MobileNetV3 TL + Aug"}

    def draw_stack(ax, layers, title):
        # vẽ các lớp của model thành các ô xếp chồng, input ở dưới cùng
        h, gap = 0.62, 0.32
        for i, (text, color) in enumerate(layers):
            y = i * (h + gap)
            ax.add_patch(FancyBboxPatch((0.05, y), 0.9, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                        fc=color, ec=MUTED, lw=0.8))
            ax.text(0.5, y + h / 2, text, ha="center", va="center", fontsize=8.5, color=INK)
            if i:
                ax.annotate("", (0.5, y), (0.5, y - gap), arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9))
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.3, len(layers) * (h + gap))
        ax.axis("off")
        ax.set_title(title, fontsize=10, color=INK, pad=6)

    return BLOCK, BLUE, CONV, FC, FROZEN, INK, IO, LABELS, MUTED, NAMES, ORANGE, POOL, draw_stack


@app.cell
def _(BLUE, CLASSES, DATA_DIR, FIG, INK, ORANGE, json, np, plt):
    # 1. Số ảnh mỗi lớp trước / sau làm sạch
    _s = json.loads((DATA_DIR / "cleaning_summary.json").read_text())
    _x = np.arange(len(CLASSES))
    _fig, _ax = plt.subplots(figsize=(6, 3.4))
    for _k, (_key, _label, _color) in enumerate([("raw", "Original", BLUE), ("clean", "After cleaning", ORANGE)]):
        _bars = _ax.bar(_x + (_k - 0.5) * 0.38, [_s[_key][c] for c in CLASSES], 0.36, label=_label, color=_color,
                        edgecolor="white", linewidth=2)
        _ax.bar_label(_bars, padding=2, fontsize=8, color=INK)
    _ax.set_xticks(_x, CLASSES)
    _ax.set_ylabel("Images")
    _ax.grid(axis="x", visible=False)
    _ax.set_ylim(0, 2050)
    _ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2)
    _fig.savefig(FIG / "class_distribution.png")
    _fig
    return


@app.cell
def _(CLASSES, FIG, load_split, np, plt):
    # 2. Ảnh mẫu mỗi lớp
    train_imgs, _labels = load_split("train", 224)
    _labels = np.array(_labels)
    _fig, _axes = plt.subplots(3, 5, figsize=(5 * 1.8, 3 * 1.9))
    _rng = np.random.default_rng(1)
    for _r, _c in enumerate(CLASSES):
        for _k, _i in enumerate(_rng.choice(np.where(_labels == _r)[0], 5, replace=False)):
            _ax = _axes[_r, _k]
            _ax.imshow(train_imgs[_i, 0], cmap="gray")
            _ax.set_xticks([]), _ax.set_yticks([])
            _ax.grid(False)
            if _k == 0:
                _ax.set_ylabel(_c, fontsize=11)
    _fig.savefig(FIG / "samples.png")
    _fig
    return (train_imgs,)


@app.cell
def _(FIG, MUTED, plt, torch, train_augment, train_imgs):
    # 3. Ví dụ augmentation
    _aug = train_augment()
    torch.manual_seed(0)
    _fig, _axes = plt.subplots(2, 5, figsize=(5 * 1.8, 2 * 1.9))
    for _r, _idx in enumerate([3, 1500]):
        for _k in range(5):
            _ax = _axes[_r, _k]
            _ax.imshow(train_imgs[_idx, 0] if _k == 0 else _aug(train_imgs[_idx])[0], cmap="gray", vmin=0, vmax=255)
            _ax.set_xticks([]), _ax.set_yticks([])
            _ax.grid(False)
            _ax.set_title("original" if _k == 0 else f"augmented {_k}", fontsize=9, color=MUTED)
    _fig.savefig(FIG / "augmentation.png")
    _fig
    return


@app.cell
def _(BLOCK, CONV, FC, FIG, FROZEN, FancyBboxPatch, IO, MUTED, POOL, draw_stack, mo, plt):
    # 4. Sơ đồ kiến trúc các model
    _fig, _ax = plt.subplots(figsize=(3.4, 7))
    draw_stack(_ax, [("Input 1×128×128", IO), ("3×3 Conv(32), ReLU", CONV), ("3×3 Conv(32), ReLU", CONV),
                     ("2×2 MaxPool", POOL), ("Flatten → 123,008", POOL),
                     ("FC(128), ReLU, Dropout 0.5", FC), ("FC(3), Softmax", FC)], "Simple CNN (~15.8M params)")
    _fig.savefig(FIG / "arch_simple_cnn.png")
    _simple = _fig

    _fig = plt.figure(figsize=(9.5, 8))
    draw_stack(_fig.add_axes([0.0, 0.0, 0.38, 1]), [
        ("Input 1×224×224", IO),
        ("7×7 Conv(32), stride 2, BN, ReLU", CONV), ("3×3 MaxPool, stride 2", POOL),
        ("1×1 Conv(64) → 3×3 Conv(96)", CONV), ("3×3 MaxPool, stride 2  → 96×28×28", POOL),
        ("Inception block A → 128 ch", BLOCK), ("Inception block B → 224 ch", BLOCK),
        ("3×3 MaxPool, stride 2  → 14×14", POOL),
        ("2 × Residual block (256)", BLOCK), ("2 × Residual block (384), first stride 2", BLOCK),
        ("Global AvgPool → 384", POOL),
        ("Dropout 0.4, FC(128), ReLU", FC), ("Dropout 0.3, FC(3), Softmax", FC),
    ], "Complex CNN (~7.6M params)")
    # khối Inception: 4 nhánh song song
    _ax2 = _fig.add_axes([0.42, 0.5, 0.56, 0.48])
    _ax2.axis("off")
    _ax2.set_xlim(0, 4)
    _ax2.set_ylim(0, 4.4)
    _ax2.set_title("Inception block", fontsize=10)
    _ax2.add_patch(FancyBboxPatch((1.2, 0.1), 1.6, 0.5, boxstyle="round,pad=0.02", fc=IO, ec=MUTED, lw=0.8))
    _ax2.text(2, 0.35, "Previous layer", ha="center", va="center", fontsize=8)
    for _b, _steps in enumerate([["1×1 Conv"], ["1×1 Conv", "3×3 Conv"], ["1×1 Conv", "5×5 Conv"],
                                 ["3×3 MaxPool", "1×1 Conv"]]):
        _bx = 0.05 + _b * 1.0
        for _st, _text in enumerate(_steps):
            _by = 1.0 + _st * 0.9
            _col = POOL if "Pool" in _text else CONV
            _ax2.add_patch(FancyBboxPatch((_bx, _by), 0.9, 0.5, boxstyle="round,pad=0.02", fc=_col, ec=MUTED, lw=0.8))
            _ax2.text(_bx + 0.45, _by + 0.25, _text, ha="center", va="center", fontsize=7.5)
        _ax2.annotate("", (_bx + 0.45, 1.0), (2, 0.6), arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.7))
        _top = 1.0 + (len(_steps) - 1) * 0.9 + 0.5
        _ax2.annotate("", (2, 3.5), (_bx + 0.45, _top), arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.7))
    _ax2.add_patch(FancyBboxPatch((1.2, 3.5), 1.6, 0.5, boxstyle="round,pad=0.02", fc=BLOCK, ec=MUTED, lw=0.8))
    _ax2.text(2, 3.75, "Concatenate (channels)", ha="center", va="center", fontsize=8)
    # khối Residual
    draw_stack(_fig.add_axes([0.55, 0.05, 0.3, 0.4]),
               [("x", IO), ("3×3 Conv, BN, ReLU", CONV), ("3×3 Conv, BN", CONV),
                ("+ shortcut (identity / 1×1 Conv), ReLU", BLOCK)], "Residual block")
    _fig.savefig(FIG / "arch_complex_cnn.png")
    _complex = _fig

    _layers = [("Input 3×224×224 (gray copied to RGB)", IO)]
    _layers += [(f"{t}  [frozen]", FROZEN) for t in ["Conv 3×3 stem", "Inverted-residual blocks 1-12"]]
    _layers += [(f"{t}  [fine-tuned]", CONV) for t in ["Inverted-residual blocks 13-15", "1×1 Conv(960)"]]
    _layers += [("Global AvgPool", POOL), ("Dropout + FC(3), Softmax  [new]", FC)]
    _fig, _ax = plt.subplots(figsize=(3.8, 0.55 * len(_layers) + 0.6))
    draw_stack(_ax, _layers, "MobileNetV3-Large transfer learning")
    _fig.savefig(FIG / "arch_mobilenetv3_tl.png")
    mo.hstack([_simple, _complex, _fig], align="start")
    return


@app.cell
def _(BLUE, FIG, LABELS, MUTED, NAMES, ORANGE, RESULTS, json, np, plt):
    # 5. Đường cong loss / accuracy khi train
    _fig, _axes = plt.subplots(len(NAMES), 2, figsize=(9, 2.4 * len(NAMES)), squeeze=False)
    for _r, _n in enumerate(NAMES):
        _h = json.loads((RESULTS / _n / "history.json").read_text())
        _ep = np.arange(1, len(_h["train_loss"]) + 1)
        for _c, _key in enumerate(["loss", "acc"]):
            _ax = _axes[_r, _c]
            _ax.plot(_ep, _h[f"train_{_key}"], color=BLUE, lw=2, label="train")
            _ax.plot(_ep, _h[f"val_{_key}"], color=ORANGE, lw=2, label="validation")
            if "finetune" in _h["phase"]:
                _ax.axvline(_h["phase"].index("finetune") + 0.5, color=MUTED, ls="--", lw=1)
            _ax.set_title(f"{LABELS[_n]} - {'loss' if _key == 'loss' else 'accuracy'}", fontsize=9)
            _ax.set_xlabel("epoch")
            if _r == 0 and _c == 0:
                _ax.legend(frameon=False)
    _fig.tight_layout()
    _fig.savefig(FIG / "training_curves.png")
    _fig
    return


@app.cell
def _(CLASSES, FIG, INK, LABELS, NAMES, RESULTS, json, np, plt):
    # 6. Confusion matrix trên tập test
    _fig, _axes = plt.subplots(2, 2, figsize=(4.2 * 2, 3.4 * 2), squeeze=False)
    for _ax, _n in zip(_axes.flat, NAMES):
        _cm = np.array(json.loads((RESULTS / _n / "metrics_test.json").read_text())["confusion_matrix"])
        _ax.imshow(_cm / _cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for _i in range(3):
            for _j in range(3):
                _ax.text(_j, _i, _cm[_i, _j], ha="center", va="center", fontsize=10,
                         color="white" if _cm[_i, _j] / _cm[_i].sum() > 0.5 else INK)
        _ax.set_xticks(range(3), CLASSES, fontsize=8)
        _ax.set_yticks(range(3), CLASSES, fontsize=8)
        _ax.set_xlabel("Predicted")
        _ax.set_ylabel("True")
        _ax.grid(False)
        _ax.set_title(LABELS[_n], fontsize=9)
    _fig.tight_layout()
    _fig.savefig(FIG / "confusion_matrices.png")
    _fig
    return


@app.cell
def _(CLASSES, LABELS, NAMES, RESULTS, json, mo):
    # 7. Bảng kết quả -> results/results_tables.md
    _rows = ["| Model | Accuracy | Precision | Recall | F1-score | Params | Epochs (best) | Train time (s) |",
             "|---|---|---|---|---|---|---|---|"]
    _per_class = ["| Model | Class | Precision | Recall | F1-score | Support |", "|---|---|---|---|---|---|"]
    for _n in NAMES:
        _m = json.loads((RESULTS / _n / "metrics_test.json").read_text())
        _meta = json.loads((RESULTS / _n / "meta.json").read_text())
        _rows.append(f"| {LABELS[_n]} | {_m['accuracy']:.4f} | {_m['precision_macro']:.4f} | {_m['recall_macro']:.4f} "
                     f"| {_m['f1_macro']:.4f} | {_meta['params_total'] / 1e6:.2f}M | {_meta['epochs_run']} "
                     f"({_meta['best_epoch']}) | {_meta['train_seconds']:.0f} |")
        for _cls in CLASSES:
            _r = _m["report"][_cls]
            _per_class.append(f"| {LABELS[_n]} | {_cls} | {_r['precision']:.4f} | {_r['recall']:.4f} "
                              f"| {_r['f1-score']:.4f} | {int(_r['support'])} |")
    _tune = ["| Learning rate | Batch size | Val accuracy | Val macro-F1 | Best epoch |", "|---|---|---|---|---|"]
    for _lr in (3e-3, 1e-3, 3e-4):
        for _bs in (32, 64):
            _d = RESULTS / f"tune_simple_lr{_lr:g}_bs{_bs}"
            _m = json.loads((_d / "metrics_val.json").read_text())
            _meta = json.loads((_d / "meta.json").read_text())
            _tune.append(f"| {_lr:g} | {_bs} | {_m['accuracy']:.4f} | {_m['f1_macro']:.4f} | {_meta['best_epoch']} |")
    _text = "## Test-set results (macro average)\n\n" + "\n".join(_rows)
    _text += "\n\n## Per-class test results\n\n" + "\n".join(_per_class)
    _text += "\n\n## Simple CNN hyperparameter search (validation set)\n\n" + "\n".join(_tune) + "\n"
    (RESULTS / "results_tables.md").write_text(_text, encoding="utf-8")
    mo.md(_text)
    return


@app.cell
def _(CLASSES, DEVICE, FIG, INK, LABELS, RESULTS, XrayDataset, load_split, mo, models, np, plt, torch):
    # 8. Grad-CAM: tô màu vùng ảnh mà model nhìn vào khi đoán
    _imgs, _labels = load_split("test", 224)
    _labels = np.array(_labels)
    _figs = []
    for _name, _model, _size, _rgb in models:
        if _name not in ("complex_cnn_aug", "mobilenetv3_tl"):
            continue
        for _p in _model.parameters():
            _p.requires_grad = True
        _layer = _model.residual[-1] if _name.startswith("complex") else _model.features[-1]  # lớp conv cuối

        # mỗi lớp 3 ảnh đoán đúng + tối đa 6 ảnh đoán sai
        _pred = np.load(RESULTS / _name / "probs_test.npy").argmax(1)
        _rng = np.random.default_rng(0)
        _picks = [i for c in range(3) for i in _rng.choice(np.where((_labels == c) & (_pred == c))[0], 3, replace=False)]
        _wrong = np.where(_pred != _labels)[0]
        _picks += list(_rng.choice(_wrong, min(6, len(_wrong)), replace=False))
        _ds = XrayDataset(_imgs, _labels.tolist(), augment=False, rgb=_rgb)
        _x = torch.stack([_ds[i][0] for i in _picks]).to(DEVICE)

        # feature map và gradient của lớp conv cuối -> heatmap
        _acts, _grads = {}, {}
        _h1 = _layer.register_forward_hook(lambda _m, _i, o: _acts.__setitem__("a", o))
        _h2 = _layer.register_full_backward_hook(lambda _m, _gi, go: _grads.__setitem__("g", go[0]))
        _model.zero_grad()
        _out = _model(_x)
        _cls = _out.argmax(1)
        _out.gather(1, _cls[:, None]).sum().backward()
        _h1.remove(), _h2.remove()
        _w = _grads["g"].mean((2, 3), keepdim=True)
        _cam = torch.relu((_w * _acts["a"]).sum(1, keepdim=True))
        _cam = torch.nn.functional.interpolate(_cam, _x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        _cam = (_cam / (_cam.amax((1, 2), keepdim=True) + 1e-8)).detach().cpu().numpy()
        _prob = torch.softmax(_out, 1).detach().cpu().numpy()
        _cls = _cls.cpu().numpy()
        for _p in _model.parameters():
            _p.requires_grad = False

        # vẽ heatmap lên ảnh, tiêu đề đỏ = đoán sai
        _fig, _axes = plt.subplots(int(np.ceil(len(_picks) / 6)), 6, figsize=(12, 2.5 * np.ceil(len(_picks) / 6)))
        for _k, _ax in enumerate(_axes.flat):
            _ax.axis("off")
            if _k >= len(_picks):
                continue
            _i = _picks[_k]
            _ax.imshow(_imgs[_i, 0], cmap="gray")
            _ax.imshow(_cam[_k], cmap="jet", alpha=0.35)
            _ax.set_title(f"true {CLASSES[_labels[_i]]}\npred {CLASSES[_cls[_k]]} ({_prob[_k, _cls[_k]]:.2f})",
                          fontsize=8, color=INK if _cls[_k] == _labels[_i] else "red")
        _fig.suptitle(f"Grad-CAM: {LABELS[_name]} (red = wrong prediction)", fontsize=10)
        _fig.tight_layout()
        _fig.savefig(FIG / f"gradcam_{_name}.png")
        _figs.append(_fig)
    mo.vstack(_figs)
    return


if __name__ == "__main__":
    app.run()
