# Task 4: Transfer learning với ResNet-50 và MobileNetV3 (đã train sẵn trên ImageNet)
# Chạy: uv run python src/4_transfer_learning.py
import torch

from models import mobilenet_transfer, resnet50_transfer, unfreeze_last_stages
from utils import evaluate_model, train_model

for name, build in [("resnet50_tl", resnet50_transfer), ("mobilenetv3_tl", mobilenet_transfer)]:
    # 1. Cấu hình: ảnh 224x224, nhân thành 3 kênh (rgb=True), có augmentation
    #    5 epoch đầu chỉ train lớp cuối (lr 0.001), sau đó fine-tune tầng cuối với lr 0.0001
    cfg = dict(name=name, augment=True, lr=1e-3, batch_size=32, epochs=30, patience=8,
               img_size=224, rgb=True, head_epochs=5, finetune_lr=1e-4)

    # 2. Build model: lấy model pretrained, đóng băng hết, thay lớp cuối thành 3 đầu ra
    torch.manual_seed(42)
    model = build()

    # 3. Train 2 giai đoạn (unfreeze_last_stages được gọi ở epoch thứ 6)
    model = train_model(model, cfg, unfreeze=unfreeze_last_stages)

    # 4. Đánh giá trên tập test và val
    evaluate_model(model, name, "test", rgb=True, size=224)
    evaluate_model(model, name, "val", rgb=True, size=224)
