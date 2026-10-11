# Task 3: Complex CNN (khối Inception + khối Residual)
# Chạy: uv run python src/3_complex_cnn.py
import torch

from models import ComplexCNN
from utils import evaluate_model, train_model

# 1. Cấu hình: ảnh 224x224, có augmentation
cfg = dict(name="complex_cnn_aug", augment=True, lr=1e-3, batch_size=32,
           epochs=40, patience=8, img_size=224, rgb=False)

# 2. Build model
torch.manual_seed(42)
model = ComplexCNN()

# 3. Train (lưu model tốt nhất ra results/complex_cnn_aug/best.pt)
model = train_model(model, cfg)

# 4. Đánh giá trên tập test và val
evaluate_model(model, cfg["name"], "test", rgb=False, size=224)
evaluate_model(model, cfg["name"], "val", rgb=False, size=224)
