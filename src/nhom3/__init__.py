import json
import sys

from .data import DATA_DIR, build_splits
from .figures import make_all
from .predict import predict_image, predict_test
from .train import EXPERIMENTS, TUNING, by_name, evaluate, run, train

USAGE = "Usage: nhom3 split | tune | train [name ...] | figures | predict [image.png]"


def main():
    if len(sys.argv) < 2:
        print(USAGE)
        return
    cmd = sys.argv[1]

    # làm sạch dữ liệu + chia train/val/test
    if cmd == "split":
        summary = build_splits()
        text = json.dumps(summary, indent=1)
        (DATA_DIR / "cleaning_summary.json").write_text(text)
        print(text)

    # dò lr và batch size cho simple cnn
    elif cmd == "tune":
        for cfg in TUNING:
            train(cfg)
            r = evaluate(cfg, "val")
            print(f"{cfg.name}: val acc {r['accuracy']:.4f} f1 {r['f1_macro']:.4f}")

    # train + test, không ghi tên thì chạy cả 5 model
    elif cmd == "train":
        names = sys.argv[2:]
        if names:
            configs = [by_name(n) for n in names]
        else:
            configs = EXPERIMENTS
        for cfg in configs:
            run(cfg)

    # vẽ hình cho báo cáo
    elif cmd == "figures":
        make_all()

    # dùng model đã lưu để dự đoán: không ghi gì thì chấm tập test, ghi đường dẫn ảnh thì đoán ảnh đó
    elif cmd == "predict":
        if len(sys.argv) > 2:
            predict_image(sys.argv[2])
        else:
            predict_test()

    else:
        print(USAGE)
