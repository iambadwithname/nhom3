import json
import sys


def main():
    cmd, *args = sys.argv[1:] or ["help"]

    if cmd == "split":
        from .data import DATA_DIR, build_splits

        summary = build_splits()
        (DATA_DIR / "cleaning_summary.json").write_text(json.dumps(summary, indent=1))
        print(json.dumps(summary, indent=1))

    elif cmd == "tune":
        from .train import TUNING, evaluate, train

        for cfg in TUNING:
            train(cfg)
            r = evaluate(cfg, "val")
            print(f"{cfg.name}: val acc {r['accuracy']:.4f} f1 {r['f1_macro']:.4f}")

    elif cmd == "train":
        from .train import EXPERIMENTS, by_name, run

        for cfg in [by_name(a) for a in args] or EXPERIMENTS:
            run(cfg)

    elif cmd == "figures":
        from .figures import make_all

        make_all()

    else:
        print("Usage: nhom3 split | tune | train [name ...] | figures")
