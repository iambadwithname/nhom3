import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyBboxPatch

from .data import CLASSES, DATA_DIR, XrayDataset, get_split, train_augment
from .train import DEVICE, EXPERIMENTS, RESULTS, TUNING, load_trained

FIG = RESULTS / "figures"
BLUE, ORANGE = "tab:blue", "tab:orange"
INK, MUTED = "black", "dimgray"
LABELS = {
    "simple_cnn": "Simple CNN",
    "simple_cnn_aug": "Simple CNN + Aug",
    "complex_cnn_aug": "Complex CNN + Aug",
    "resnet50_tl": "ResNet-50 TL + Aug",
    "mobilenetv3_tl": "MobileNetV3 TL + Aug",
}

plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3, "savefig.dpi": 200, "savefig.bbox": "tight"})


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name)
    plt.close(fig)


def class_distribution():
    s = json.loads((DATA_DIR / "cleaning_summary.json").read_text())
    x = np.arange(len(CLASSES))
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for k, (key, label, color) in enumerate([("raw", "Original", BLUE), ("clean", "After cleaning", ORANGE)]):
        vals = [s[key][c] for c in CLASSES]
        bars = ax.bar(x + (k - 0.5) * 0.38, vals, 0.36, label=label, color=color, edgecolor="white", linewidth=2)
        ax.bar_label(bars, padding=2, fontsize=8, color=INK)
    ax.set_xticks(x, CLASSES)
    ax.set_ylabel("Images")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 2050)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2)
    _save(fig, "class_distribution.png")


def sample_grid(n=5):
    imgs, labels = get_split("train")
    labels = np.array(labels)
    fig, axes = plt.subplots(3, n, figsize=(n * 1.8, 3 * 1.9))
    rng = np.random.default_rng(1)
    for r, c in enumerate(CLASSES):
        for k, i in enumerate(rng.choice(np.where(labels == r)[0], n, replace=False)):
            ax = axes[r, k]
            ax.imshow(imgs[i, 0], cmap="gray")
            ax.set_xticks([]), ax.set_yticks([])
            ax.grid(False)
            if k == 0:
                ax.set_ylabel(c, fontsize=11)
    _save(fig, "samples.png")


def augmentation_examples(n=5):
    imgs, _ = get_split("train")
    aug = train_augment()
    torch.manual_seed(0)
    fig, axes = plt.subplots(2, n, figsize=(n * 1.8, 2 * 1.9))
    for r, idx in enumerate([3, 1500]):
        for k in range(n):
            ax = axes[r, k]
            ax.imshow(imgs[idx, 0] if k == 0 else aug(imgs[idx])[0], cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([]), ax.set_yticks([])
            ax.grid(False)
            ax.set_title("original" if k == 0 else f"augmented {k}", fontsize=9, color=MUTED)
    _save(fig, "augmentation.png")


def _stack(ax, layers, title):
    # vẽ các lớp thành ô xếp chồng, input ở dưới
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


CONV, POOL, FC, IO, BLOCK, FROZEN = "lightblue", "peachpuff", "palegreen", "whitesmoke", "lavender", "lightgray"


def architectures():
    simple = [
        ("Input 1×128×128", IO),
        ("3×3 Conv(32), ReLU", CONV),
        ("3×3 Conv(32), ReLU", CONV),
        ("2×2 MaxPool", POOL),
        ("Flatten → 123,008", POOL),
        ("FC(128), ReLU, Dropout 0.5", FC), ("FC(3), Softmax", FC),
    ]
    fig, ax = plt.subplots(figsize=(3.4, 7))
    _stack(ax, simple, "Simple CNN (~15.8M params)")
    _save(fig, "arch_simple_cnn.png")

    complex_ = [
        ("Input 1×224×224", IO),
        ("7×7 Conv(32), stride 2, BN, ReLU", CONV), ("3×3 MaxPool, stride 2", POOL),
        ("1×1 Conv(64) → 3×3 Conv(96)", CONV), ("3×3 MaxPool, stride 2  → 96×28×28", POOL),
        ("Inception block A → 128 ch", BLOCK), ("Inception block B → 224 ch", BLOCK),
        ("3×3 MaxPool, stride 2  → 14×14", POOL),
        ("2 × Residual block (256)", BLOCK), ("2 × Residual block (384), first stride 2", BLOCK),
        ("Global AvgPool → 384", POOL),
        ("Dropout 0.4, FC(128), ReLU", FC), ("Dropout 0.3, FC(3), Softmax", FC),
    ]
    fig = plt.figure(figsize=(9.5, 8))
    ax = fig.add_axes([0.0, 0.0, 0.38, 1])
    _stack(ax, complex_, "Complex CNN (~7.6M params)")

    ax2 = fig.add_axes([0.42, 0.5, 0.56, 0.48])
    ax2.axis("off")
    ax2.set_xlim(0, 4)
    ax2.set_ylim(0, 4.4)
    ax2.set_title("Inception block", fontsize=10)
    branches = [["1×1 Conv"], ["1×1 Conv", "3×3 Conv"], ["1×1 Conv", "5×5 Conv"], ["3×3 MaxPool", "1×1 Conv"]]
    ax2.add_patch(FancyBboxPatch((1.2, 0.1), 1.6, 0.5, boxstyle="round,pad=0.02", fc=IO, ec=MUTED, lw=0.8))
    ax2.text(2, 0.35, "Previous layer", ha="center", va="center", fontsize=8)
    for b, steps in enumerate(branches):
        x = 0.05 + b * 1.0
        for s, name in enumerate(steps):
            y = 1.0 + s * 0.9
            col = POOL if "Pool" in name else CONV
            ax2.add_patch(FancyBboxPatch((x, y), 0.9, 0.5, boxstyle="round,pad=0.02", fc=col, ec=MUTED, lw=0.8))
            ax2.text(x + 0.45, y + 0.25, name, ha="center", va="center", fontsize=7.5)
        ax2.annotate("", (x + 0.45, 1.0), (2, 0.6), arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.7))
        top = 1.0 + (len(steps) - 1) * 0.9 + 0.5
        ax2.annotate("", (2, 3.5), (x + 0.45, top), arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.7))
    ax2.add_patch(FancyBboxPatch((1.2, 3.5), 1.6, 0.5, boxstyle="round,pad=0.02", fc=BLOCK, ec=MUTED, lw=0.8))
    ax2.text(2, 3.75, "Concatenate (channels)", ha="center", va="center", fontsize=8)

    ax3 = fig.add_axes([0.55, 0.05, 0.3, 0.4])
    _stack(ax3, [("x", IO), ("3×3 Conv, BN, ReLU", CONV), ("3×3 Conv, BN", CONV),
                 ("+ shortcut (identity / 1×1 Conv), ReLU", BLOCK)], "Residual block")
    _save(fig, "arch_complex_cnn.png")

    for name, title, frozen, trainable in [
        ("arch_resnet50_tl.png", "ResNet-50 transfer learning",
         ["conv1 7×7 + MaxPool", "layer1 (3 bottlenecks)", "layer2 (4 bottlenecks)", "layer3 (6 bottlenecks)"],
         ["layer4 (3 bottlenecks)"]),
        ("arch_mobilenetv3_tl.png", "MobileNetV3-Large transfer learning",
         ["Conv 3×3 stem", "Inverted-residual blocks 1-12"],
         ["Inverted-residual blocks 13-15", "1×1 Conv(960)"]),
    ]:
        head = [("Global AvgPool", POOL), ("Dropout + FC(3), Softmax  [new]", FC)]
        layers = [("Input 3×224×224 (gray copied to RGB)", IO)]
        layers += [(f"{t}  [frozen]", FROZEN) for t in frozen]
        layers += [(f"{t}  [fine-tuned]", CONV) for t in trainable] + head
        fig, ax = plt.subplots(figsize=(3.8, 0.55 * len(layers) + 0.6))
        _stack(ax, layers, title)
        _save(fig, name)


def curves():
    names = [c.name for c in EXPERIMENTS if (RESULTS / c.name / "history.json").exists()]
    fig, axes = plt.subplots(len(names), 2, figsize=(9, 2.4 * len(names)), squeeze=False)
    for r, n in enumerate(names):
        h = json.loads((RESULTS / n / "history.json").read_text())
        ep = np.arange(1, len(h["train_loss"]) + 1)
        for c, key in enumerate(["loss", "acc"]):
            ax = axes[r, c]
            ax.plot(ep, h[f"train_{key}"], color=BLUE, lw=2, label="train")
            ax.plot(ep, h[f"val_{key}"], color=ORANGE, lw=2, label="validation")
            if "finetune" in h["phase"]:
                ax.axvline(h["phase"].index("finetune") + 0.5, color=MUTED, ls="--", lw=1)
            ax.set_title(f"{LABELS[n]} - {'loss' if key == 'loss' else 'accuracy'}", fontsize=9)
            ax.set_xlabel("epoch")
            if r == 0 and c == 0:
                ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, "training_curves.png")


def confusion_matrices():
    names = [c.name for c in EXPERIMENTS if (RESULTS / c.name / "metrics_test.json").exists()]
    cols = 3
    rows = (len(names) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.4 * rows), squeeze=False)
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    for ax, n in zip(axes.flat, names):
        cm = np.array(json.loads((RESULTS / n / "metrics_test.json").read_text())["confusion_matrix"])
        ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                frac = cm[i, j] / cm[i].sum()
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=10,
                        color="white" if frac > 0.5 else INK)
        ax.set_xticks(range(3), CLASSES, fontsize=8)
        ax.set_yticks(range(3), CLASSES, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.grid(False)
        ax.set_title(LABELS[n], fontsize=9)
    fig.tight_layout()
    _save(fig, "confusion_matrices.png")


def tables():
    rows = ["| Model | Accuracy | Precision | Recall | F1-score | Params | Epochs (best) | Train time (s) |",
            "|---|---|---|---|---|---|---|---|"]
    per_class = ["| Model | Class | Precision | Recall | F1-score | Support |", "|---|---|---|---|---|---|"]
    for c in EXPERIMENTS:
        d = RESULTS / c.name
        if not (d / "metrics_test.json").exists():
            continue
        m = json.loads((d / "metrics_test.json").read_text())
        meta = json.loads((d / "meta.json").read_text())
        rows.append(
            f"| {LABELS[c.name]} | {m['accuracy']:.4f} | {m['precision_macro']:.4f} | {m['recall_macro']:.4f} "
            f"| {m['f1_macro']:.4f} | {meta['params_total'] / 1e6:.2f}M | {meta['epochs_run']} ({meta['best_epoch']}) "
            f"| {meta['train_seconds']:.0f} |"
        )
        for cls in CLASSES:
            r = m["report"][cls]
            per_class.append(f"| {LABELS[c.name]} | {cls} | {r['precision']:.4f} | {r['recall']:.4f} "
                             f"| {r['f1-score']:.4f} | {int(r['support'])} |")
    tune = ["| Learning rate | Batch size | Val accuracy | Val macro-F1 | Best epoch |", "|---|---|---|---|---|"]
    for c in TUNING:
        d = RESULTS / c.name
        if (d / "metrics_val.json").exists():
            m = json.loads((d / "metrics_val.json").read_text())
            meta = json.loads((d / "meta.json").read_text())
            tune.append(f"| {c.lr:g} | {c.batch_size} | {m['accuracy']:.4f} | {m['f1_macro']:.4f} | {meta['best_epoch']} |")
    text = "## Test-set results (macro average)\n\n" + "\n".join(rows)
    text += "\n\n## Per-class test results\n\n" + "\n".join(per_class)
    text += "\n\n## Simple CNN hyperparameter search (validation set)\n\n" + "\n".join(tune) + "\n"
    (RESULTS / "results_tables.md").write_text(text, encoding="utf-8")


def _target_layer(model, name):
    if name.startswith("simple"):
        return model.features[3]
    if name.startswith("complex"):
        return model.residual[-1]
    if name.startswith("resnet"):
        return model.layer4
    return model.features[-1]


def gradcam(model, layer, x, cls=None):
    acts, grads = {}, {}
    h1 = layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
    model.zero_grad()
    out = model(x)
    cls = out.argmax(1) if cls is None else cls
    out.gather(1, cls[:, None]).sum().backward()
    h1.remove(), h2.remove()
    w = grads["g"].mean((2, 3), keepdim=True)
    cam = torch.relu((w * acts["a"]).sum(1, keepdim=True))
    cam = torch.nn.functional.interpolate(cam, x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    cam = cam / (cam.amax((1, 2), keepdim=True) + 1e-8)
    return cam.detach().cpu().numpy(), cls.cpu().numpy(), torch.softmax(out, 1).detach().cpu().numpy()


def gradcam_figures(names=("complex_cnn_aug", "resnet50_tl")):
    imgs, labels = get_split("test")
    labels = np.array(labels)
    for name in names:
        cfg = next(c for c in EXPERIMENTS if c.name == name)
        if not (RESULTS / name / "best.pt").exists():
            continue
        model = load_trained(cfg)
        for p in model.parameters():
            p.requires_grad = True
        ds = XrayDataset(imgs, labels.tolist(), augment=False, rgb=cfg.rgb)
        prob = np.load(RESULTS / name / "probs_test.npy")
        pred = prob.argmax(1)
        rng = np.random.default_rng(0)
        # mỗi lớp 3 ảnh đoán đúng + tối đa 6 ảnh đoán sai
        picks = [i for c in range(3) for i in rng.choice(np.where((labels == c) & (pred == c))[0], 3, replace=False)]
        wrong = np.where(pred != labels)[0]
        picks += list(rng.choice(wrong, min(6, len(wrong)), replace=False))
        x = torch.stack([ds[i][0] for i in picks]).to(DEVICE)
        cam, cls, p = gradcam(model, _target_layer(model, name), x)
        cols = 6
        rows = int(np.ceil(len(picks) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.5))
        for k, ax in enumerate(axes.flat):
            ax.axis("off")
            if k >= len(picks):
                continue
            i = picks[k]
            ax.imshow(imgs[i, 0], cmap="gray")
            ax.imshow(cam[k], cmap="jet", alpha=0.35)
            ok = cls[k] == labels[i]
            ax.set_title(f"true {CLASSES[labels[i]]}\npred {CLASSES[cls[k]]} ({p[k, cls[k]]:.2f})",
                         fontsize=8, color=INK if ok else "red")
        fig.suptitle(f"Grad-CAM: {LABELS[name]} (red = wrong prediction)", fontsize=10)
        fig.tight_layout()
        _save(fig, f"gradcam_{name}.png")


def make_all():
    class_distribution()
    sample_grid()
    augmentation_examples()
    architectures()
    curves()
    confusion_matrices()
    tables()
    gradcam_figures()
    print("figures written to", FIG)
