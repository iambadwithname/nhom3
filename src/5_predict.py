import sys

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from models import ComplexCNN, SimpleCNN, mobilenet_transfer
from utils import CLASSES, DEVICE, RESULTS, XrayDataset, predict

MODELS = [
    ("simple_cnn", SimpleCNN(img_size=128), 128, False),
    ("simple_cnn_aug", SimpleCNN(img_size=128), 128, False),
    ("complex_cnn_aug", ComplexCNN(), 224, False),
    ("mobilenetv3_tl", mobilenet_transfer(), 224, True),
]

loaded = []
for name, model, size, rgb in MODELS:
    path = RESULTS / name / "best.pt"
    if not path.exists():
        print(f"{name}: no best.pt, train this model first")
        continue
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    loaded.append((name, model.to(DEVICE).eval(), size, rgb))

if len(sys.argv) > 1:
    print(f"Image: {sys.argv[1]}")
    for name, model, size, rgb in loaded:
        img = Image.open(sys.argv[1]).convert("L").resize((size, size), Image.BILINEAR)
        x = torch.from_numpy(np.array(img))[None, None]
        x = XrayDataset(x, [0], augment=False, rgb=rgb)[0][0][None].to(DEVICE)
        with torch.no_grad():
            prob = torch.softmax(model(x).float(), 1)[0].cpu().numpy()
        k = prob.argmax()
        print(f"{name:<18} -> {CLASSES[k]:<10} ({prob[k]:.2%})")
else:
    print(f"{'Model':<18} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    matrices = []
    for name, model, size, rgb in loaded:
        y, prob = predict(model, "test", rgb, size)
        pred = prob.argmax(1)
        p, r, f, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
        print(f"{name:<18} {accuracy_score(y, pred):>9.4f} {p:>10.4f} {r:>8.4f} {f:>8.4f}")
        matrices.append((name, confusion_matrix(y, pred)))

    for name, cm in matrices:
        print(f"\nConfusion matrix - {name}")
        print(f"{'':<12}" + "".join(f"{c:>11}" for c in CLASSES))
        for c, row in zip(CLASSES, cm):
            print(f"{c:<12}" + "".join(f"{v:>11}" for v in row))
