import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from .data import CLASSES, XrayDataset
from .train import DEVICE, EXPERIMENTS, RESULTS, load_trained, predict


def predict_test():
    # đọc model đã lưu (best.pt) rồi dự đoán trên tập test, không train lại
    print(f"{'Model':<18} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    matrices = {}
    for cfg in EXPERIMENTS:
        if not (RESULTS / cfg.name / "best.pt").exists():
            print(f"{cfg.name:<18} no best.pt, run 'uv run nhom3 train {cfg.name}' first")
            continue
        model = load_trained(cfg)
        y, prob = predict(model, "test", cfg.rgb, cfg.img_size)
        pred = prob.argmax(1)
        p, r, f, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
        print(f"{cfg.name:<18} {accuracy_score(y, pred):>9.4f} {p:>10.4f} {r:>8.4f} {f:>8.4f}")
        matrices[cfg.name] = confusion_matrix(y, pred)

    # in confusion matrix của từng model (hàng = nhãn thật, cột = nhãn dự đoán)
    for name, cm in matrices.items():
        print(f"\nConfusion matrix - {name}")
        print(f"{'':<12}" + "".join(f"{c:>11}" for c in CLASSES))
        for c, row in zip(CLASSES, cm):
            print(f"{c:<12}" + "".join(f"{v:>11}" for v in row))


def predict_image(path):
    # dự đoán 1 ảnh X-quang bất kỳ bằng cả 5 model
    print(f"Image: {path}")
    for cfg in EXPERIMENTS:
        if not (RESULTS / cfg.name / "best.pt").exists():
            continue
        img = Image.open(path).convert("L").resize((cfg.img_size, cfg.img_size), Image.BILINEAR)
        x = torch.from_numpy(np.array(img))[None, None]  # 1 ảnh, 1 kênh
        x = XrayDataset(x, [0], augment=False, rgb=cfg.rgb)[0][0][None].to(DEVICE)
        model = load_trained(cfg)
        with torch.no_grad():
            prob = torch.softmax(model(x).float(), 1)[0].cpu().numpy()
        k = prob.argmax()
        print(f"{cfg.name:<18} -> {CLASSES[k]:<10} ({prob[k]:.2%})")
