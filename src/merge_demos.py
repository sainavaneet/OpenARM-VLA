from __future__ import annotations
from pathlib import Path

import h5py
import hydra
from omegaconf import DictConfig


def copy_group(src: h5py.Group, dst: h5py.Group) -> None:
    for key, item in src.items():
        if isinstance(item, h5py.Dataset):
            dst.create_dataset(key, data=item[...], dtype=item.dtype)
        else:
            copy_group(item, dst.create_group(key))


def _task_key(key: str) -> int:
    if key.startswith("task"):
        num = key[4:]
        return int(num) if num.isdigit() else 10**9
    return 10**9


@hydra.main(version_base=None, config_path="../conf", config_name="merge_demos")
def main(cfg: DictConfig) -> None:
    if "root" not in cfg or "out" not in cfg:
        raise ValueError("Config must define 'root' and 'out' for merge_demos.")

    root = Path(cfg.root)
    out_root = Path(cfg.out)
    out_root.mkdir(parents=True, exist_ok=True)

    if "tasks" not in cfg:
        raise ValueError("Config must define 'tasks' as a map like task0: name0.")

    tasks = cfg.tasks
    if not isinstance(tasks, dict):
        raise ValueError("'tasks' must be a map/dict like task0: name0.")

    task_names = [tasks[k] for k in sorted(tasks.keys(), key=_task_key)]
    for pose_idx, task_name in enumerate(task_names):
        pose_dir = root / f"pose{pose_idx}"
        if pose_dir.is_dir():
            demos = sorted(pose_dir.glob(f"demo_pose{pose_idx}_*.hdf5"))
        else:
            demos = sorted(root.glob(f"demo_pose{pose_idx}_*.hdf5"))
        if "num_demos" in cfg and cfg.num_demos is not None:
            demos = demos[: cfg.num_demos]
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
