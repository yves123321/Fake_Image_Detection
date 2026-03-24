import os
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--out_md", type=str, required=True)
    parser.add_argument("--out_tex", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(os.path.expanduser(args.csv))
    df = df.sort_values(["model", "img_size", "data_frac"]).reset_index(drop=True)

    # resolution table: only data_frac == 1.0
    res_df = df[df["data_frac"] == 1.0][[
        "model", "img_size", "fid_ai", "fid_nature", "consistency_ai", "consistency_nature"
    ]].copy()

    # data scale table: only img_size == 64
    scale_df = df[df["img_size"] == 64][[
        "model", "data_frac", "fid_ai", "fid_nature", "consistency_ai", "consistency_nature"
    ]].copy()

    with open(os.path.expanduser(args.out_md), "w") as f:
        f.write("# Resolution sweep\n\n")
        f.write(res_df.to_markdown(index=False))
        f.write("\n\n# Data scale sweep\n\n")
        f.write(scale_df.to_markdown(index=False))
        f.write("\n")

    with open(os.path.expanduser(args.out_tex), "w") as f:
        f.write("% Resolution sweep\n")
        f.write(res_df.to_latex(index=False, float_format="%.3f"))
        f.write("\n\n% Data scale sweep\n")
        f.write(scale_df.to_latex(index=False, float_format="%.3f"))

    print("saved:", args.out_md)
    print("saved:", args.out_tex)


if __name__ == "__main__":
    main()