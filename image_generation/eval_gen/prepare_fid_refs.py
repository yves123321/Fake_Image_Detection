import os
import argparse
from pathlib import Path
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, required=True)
    parser.add_argument("--dst_dir", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--format", type=str, default="png", choices=["png", "jpg"])
    args = parser.parse_args()

    src_dir = Path(os.path.expanduser(args.src_dir))
    dst_dir = Path(os.path.expanduser(args.dst_dir))
    dst_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG", ".JPG", ".PNG"}
    files = [p for p in src_dir.iterdir() if p.is_file() and p.suffix in exts]

    print("num input files:", len(files))

    saved = 0
    for i, p in enumerate(files):
        try:
            img = Image.open(p).convert("RGB")
            img = img.resize((args.img_size, args.img_size), Image.BILINEAR)

            if args.format == "png":
                out_path = dst_dir / f"{p.stem}.png"
                img.save(out_path, format="PNG")
            else:
                out_path = dst_dir / f"{p.stem}.jpg"
                img.save(out_path, format="JPEG", quality=95)

            saved += 1

            if (i + 1) % 500 == 0:
                print(f"processed {i+1}/{len(files)}")
        except Exception as e:
            print("skip:", p.name, e)

    print("saved:", saved)
    print("output dir:", dst_dir)


if __name__ == "__main__":
    main()