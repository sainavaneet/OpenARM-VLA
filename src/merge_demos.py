from __future__ import annotations

import argparse
from pathlib import Path

import h5py


def copy_group(src: h5py.Group, dst: h5py.Group) -> None:
    for key, item in src.items():
        if isinstance(item, h5py.Dataset):
            dst.create_dataset(key, data=item[...], dtype=item.dtype)
        else:
            copy_group(item, dst.create_group(key))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Folder with demo_<color>_*.hdf5 files")
    parser.add_argument("--out", type=str, required=True, help="Output folder")
    parser.add_argument("--num_demos", type=int, default=None)
    args = parser.parse_args()

    root = Path(args.root)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    color_to_name = {
        "red": "pick_the_red_cube_from_infront_of_the_robot",
        "green": "pick_the_green_cube_from_slightly_near_to_the_right_side_of_the_table",
    }

    for color, task_name in color_to_name.items():
        demos = sorted(root.glob(f"demo_{color}_*.hdf5"))
        if args.num_demos is not None:
            demos = demos[: args.num_demos]
        if not demos:
            print(f"---->No demos for {color}")
            continue

        out_path = out_root / f"{task_name}.hdf5"
        with h5py.File(out_path, "w") as out_f:
            data_grp = out_f.create_group("data")
            for idx, demo_path in enumerate(demos):
                with h5py.File(demo_path, "r") as src_f:
                    src_demo_key = next(iter(src_f["data"].keys()))
                    src_demo = src_f["data"][src_demo_key]
                    copy_group(src_demo, data_grp.create_group(f"demo_{idx}"))
            out_f.attrs["task"] = task_name
            out_f.attrs["color"] = color
            out_f.attrs["num_demos"] = len(demos)

        print(f"saved -----> {out_path} (demos={len(demos)})")


if __name__ == "__main__":
    main()
