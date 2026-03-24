import os
import re
import json
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict


PAT = re.compile(r"^VQDM_(\d+)_(\d+)_(\d+)_(\d{3})_vqdm_(\d{5})\.png$", re.I)


def parse_cls_id(name):
    m = PAT.match(name)
    if m is None:
        return None
    return m.group(4)  # 000~999


def safe_link_or_copy(src, dst):
    src = Path(src)
    dst = Path(dst)

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", type=str, required=True,
                        help="e.g. ~/data/vqdm_subset_raw")
    parser.add_argument("--out_root", type=str, required=True,
                        help="e.g. ~/data/vqdm_ai10")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--class_ids", type=str, default="",
                        help="comma-separated class ids like 000,001,002")
    parser.add_argument("--select_mode", type=str, default="first",
                        choices=["first", "random"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_per_class", type=int, default=0,
                        help="0 means use all available")
    parser.add_argument("--val_per_class", type=int, default=0,
                        help="0 means use all available")
    args = parser.parse_args()

    random.seed(args.seed)

    src_root = Path(os.path.expanduser(args.src_root))
    out_root = Path(os.path.expanduser(args.out_root))

    train_ai = src_root / "train" / "ai"
    val_ai = src_root / "val" / "ai"

    train_groups = defaultdict(list)
    val_groups = defaultdict(list)

    for p in train_ai.iterdir():
        if p.is_file():
            cls_id = parse_cls_id(p.name)
            if cls_id is not None:
                train_groups[cls_id].append(p)

    for p in val_ai.iterdir():
        if p.is_file():
            cls_id = parse_cls_id(p.name)
            if cls_id is not None:
                val_groups[cls_id].append(p)

    common_classes = sorted(set(train_groups.keys()) & set(val_groups.keys()))
    print("num common classes:", len(common_classes))

    if args.class_ids.strip():
        selected = [x.strip() for x in args.class_ids.split(",") if x.strip()]
    else:
        if args.select_mode == "first":
            selected = common_classes[:args.num_classes]
        else:
            selected = sorted(random.sample(common_classes, args.num_classes))

    print("selected classes:", selected)

    # 建目录
    for split in ["train", "val"]:
        for cls_id in selected:
            (out_root / split / cls_id).mkdir(parents=True, exist_ok=True)

    summary = {
        "selected_classes": selected,
        "train_count": {},
        "val_count": {}
    }

    for cls_id in selected:
        train_files = sorted(train_groups[cls_id])
        val_files = sorted(val_groups[cls_id])

        if args.train_per_class > 0:
            train_files = train_files[:args.train_per_class]
        if args.val_per_class > 0:
            val_files = val_files[:args.val_per_class]

        for p in train_files:
            safe_link_or_copy(p, out_root / "train" / cls_id / p.name)
        for p in val_files:
            safe_link_or_copy(p, out_root / "val" / cls_id / p.name)

        summary["train_count"][cls_id] = len(train_files)
        summary["val_count"][cls_id] = len(val_files)

    with open(out_root / "meta.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("done.")
    print("output:", out_root)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()