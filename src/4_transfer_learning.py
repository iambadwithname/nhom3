import torch

from models import mobilenet_transfer, unfreeze_last_stages
from utils import evaluate_model, train_model

cfg = dict(name="mobilenetv3_tl", augment=True, lr=1e-3, batch_size=32, epochs=30, patience=8,
           img_size=224, rgb=True, head_epochs=5, finetune_lr=1e-4)

torch.manual_seed(42)
model = mobilenet_transfer()

model = train_model(model, cfg, unfreeze=unfreeze_last_stages)

evaluate_model(model, cfg["name"], "test", rgb=True, size=224)
evaluate_model(model, cfg["name"], "val", rgb=True, size=224)
