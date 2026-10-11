# Task 2: Simple CNN nhóm tự thiết kế
# Chạy: uv run python src/2_simple_cnn.py
import torch

from models import SimpleCNN
from utils import evaluate_model, train_model

# 1. Dò learning rate và batch size (chấm trên tập val, không đụng tập test)
for lr in (3e-3, 1e-3, 3e-4):
    for bs in (32, 64):
        cfg = dict(name=f"tune_simple_lr{lr:g}_bs{bs}", augment=True, lr=lr, batch_size=bs,
                   epochs=25, patience=6, img_size=128, rgb=False)
        torch.manual_seed(42)
        model = SimpleCNN(img_size=128)
        model = train_model(model, cfg)
        evaluate_model(model, cfg["name"], "val", rgb=False, size=128)

# 2. Train model chính với lr = 0.001, batch size = 64 (tốt nhất ở bước 1)
#    train 2 lần: không augmentation và có augmentation để so sánh
for name, augment in [("simple_cnn", False), ("simple_cnn_aug", True)]:
    cfg = dict(name=name, augment=augment, lr=1e-3, batch_size=64,
               epochs=40, patience=8, img_size=128, rgb=False)
    torch.manual_seed(42)
    model = SimpleCNN(img_size=128)
    model = train_model(model, cfg)

    # 3. Đánh giá trên tập test và val
    evaluate_model(model, name, "test", rgb=False, size=128)
    evaluate_model(model, name, "val", rgb=False, size=128)
