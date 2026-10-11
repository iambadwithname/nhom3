# how to name my file idk
import hashlib
import os

from torch import nn


def get_dir(model: nn.Module):  # why is this fuction not in notebooks? idk
    name = str(model)
    dir = f"./model/{hashlib.sha3_256(name.encode()).hexdigest()}"
    if not os.path.isdir(dir):
        os.makedirs((dir), exist_ok=True)
        with open(f"{dir}/model.md", "w") as f:
            f.write(name)
    return dir
