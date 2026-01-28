from __future__ import annotations

import argparse
import json
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
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Folder with demo_pose<idx>_*.hdf5 files or pose<idx>/ subfolders",
    )
    parser.add_argument("--out", type=str, required=True, help="Output folder")
    parser.add_argument("--num_demos", type=int, default=None)
    parser.add_argument(
        "--task_names",
        type=str,
        default="pose0_task;pose1_task",
        help="Semicolon-separated task names for pose indices (e.g., name0;name1).",
    )
    parser.add_argument(
        "--task_map_json",
        type=str,
        default=None,
        help='JSON map like {"pose0_task":"name0","pose1_task":"name1"}',
    )
    args = parser.parse_args()

    root = Path(args.root)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.task_map_json:
        try:
            task_map = json.loads(args.task_map_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid --task_map_json: {exc}") from exc
        def _pose_key(key: str) -> int:
            if key.startswith("pose") and key.endswith("_task"):
                num = key[4:-5]
                return int(num) if num.isdigit() else 10**9
            return 10**9

        task_names = [task_map[k] for k in sorted(task_map.keys(), key=_pose_key)]
    else:
        task_names = [n.strip() for n in args.task_names.split(";") if n.strip()]
    for pose_idx, task_name in enumerate(task_names):
        pose_dir = root / f"pose{pose_idx}"
        if pose_dir.is_dir():
            demos = sorted(pose_dir.glob(f"demo_pose{pose_idx}_*.hdf5"))
        else:
            demos = sorted(root.glob(f"demo_pose{pose_idx}_*.hdf5"))
        if args.num_demos is not None:
            demos = demos[: args.num_demos]
        if not demos:
            print(f"---->No demos for pose{pose_idx}")
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
            out_f.attrs["pose_index"] = pose_idx
            out_f.attrs["num_demos"] = len(demos)

        print(f"saved -----> {out_path} (demos={len(demos)})")


if __name__ == "__main__":
    main()
