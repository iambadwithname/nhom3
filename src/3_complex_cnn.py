import torch

from models import ComplexCNN
from utils import evaluate_model, train_model

cfg = dict(name="complex_cnn_aug", augment=True, lr=1e-3, batch_size=32,
           epochs=40, patience=8, img_size=224, rgb=False)

torch.manual_seed(42)
model = ComplexCNN()

model = train_model(model, cfg)

evaluate_model(model, cfg["name"], "test", rgb=False, size=224)
evaluate_model(model, cfg["name"], "val", rgb=False, size=224)
