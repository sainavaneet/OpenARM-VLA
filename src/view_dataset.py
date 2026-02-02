from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog

try:
    from PIL import Image, ImageTk
except Exception as exc:  # pragma: no cover - dependency check
    raise SystemExit("Pillow is required for Tk image display. Install `pillow`.") from exc


def _load_demo(path: Path, demo_key: str | None = None):
    with h5py.File(path, "r") as f:
        data_grp = f["data"]
        if demo_key is None:
            demo_key = next(iter(data_grp.keys()))
        demo = data_grp[demo_key]
        obs = demo["obs"]
        cam_fixed = obs["agentview_rgb"][...]
        cam_link0 = obs["eye_in_hand_rgb"][...]
        rewards = demo["rewards"][...]
        actions = demo["actions"][...]
    return cam_link0, cam_fixed, rewards, actions


def _to_photo(img: np.ndarray) -> ImageTk.PhotoImage:
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    else:
        img = img if img.shape[-1] == 3 else img
    pil = Image.fromarray(img)
    return ImageTk.PhotoImage(pil)


def main():
    parser = argparse.ArgumentParser(description="View HDF5 demos with Tkinter + OpenCV.")
    parser.add_argument(
        "--root",
        type=str,
        default="/home/navaneet/Documents/openarm/openarm_isaac_lab/sim_transfer_cube/datasets/pickt.the.cube",
    )
    parser.add_argument("--demo", type=str, default="", help="Demo key (e.g. demo_blue_0).")
    args = parser.parse_args()

    root = tk.Tk()
    root.title("Dataset Viewer")

    state = {
        "demo": args.demo or "",
        "idx": 0,
        "path": None,
        "cam_link0": None,
        "cam_fixed": None,
        "rewards": None,
        "actions": None,
        "photo1": None,
        "photo2": None,
    }

    root_var = tk.StringVar(value=args.root)
    file_var = tk.StringVar(value="")
    demo_var = tk.StringVar(value="")

    def _demo_path(file_name: str) -> Path:
        root_path = Path(root_var.get())
        if root_path.is_file():
            return root_path
        return root_path / file_name

    def _list_files() -> list[str]:
        root_path = Path(root_var.get())
        if root_path.is_file():
            return [root_path.name]
        return [p.name for p in sorted(root_path.glob("*.hdf5"))]

    def _list_demo_keys(file_name: str) -> list[str]:
        path = _demo_path(file_name)
        if not path.exists():
            return []
        try:
            with h5py.File(path, "r") as f:
                return sorted(list(f["data"].keys()))
        except Exception:
            return []

    def _load(file_name: str, demo_key: str | None) -> bool:
        path = _demo_path(file_name)
        if not path.exists():
            return False
        cam_link0, cam_fixed, rewards, actions = _load_demo(path, demo_key)
        state.update(
            {
                "demo": demo_key or "",
                "idx": 0,
                "path": path,
                "cam_link0": cam_link0,
                "cam_fixed": cam_fixed,
                "rewards": rewards,
                "actions": actions,
            }
        )
        _update()
        return True

    def _refresh_files(select_file: str | None = None):
        file_list.delete(0, tk.END)
        files = _list_files()
        for f in files:
            file_list.insert(tk.END, f)
        if files:
            target = select_file if select_file is not None else files[0]
            if target in files:
                idx = files.index(target)
                file_list.selection_set(idx)
                file_list.activate(idx)
                _refresh_demos(target)
        else:
            _clear_state("No files found.")

    def _refresh_demos(file_name: str, select_demo: str | None = None):
        demo_list.delete(0, tk.END)
        demos = _list_demo_keys(file_name)
        for d in demos:
            demo_list.insert(tk.END, d)
        if demos:
            target = select_demo if select_demo is not None else demos[0]
            if target in demos:
                idx = demos.index(target)
                demo_list.selection_set(idx)
                demo_list.activate(idx)
                _load(file_name, target)
        else:
            _clear_state("No demos found in selected file.")

    def _clear_state(msg: str):
        state.update(
            {
                "demo": "",
                "idx": 0,
                "path": None,
                "cam_link0": None,
                "cam_fixed": None,
                "rewards": None,
                "actions": None,
                "photo1": None,
                "photo2": None,
            }
        )
        label1.configure(image="")
        label2.configure(image="")
        status.configure(text=msg)

    def _update():
        cam_link0 = state["cam_link0"]
        cam_fixed = state["cam_fixed"]
        if cam_link0 is None:
            return
        i = state["idx"]
        i = max(0, min(i, cam_link0.shape[0] - 1))
        state["idx"] = i
        frame_slider.configure(to=max(0, cam_link0.shape[0] - 1))
        if frame_slider.get() != i:
            frame_slider.set(i)

        img1 = cam_link0[i]
        img2 = cam_fixed[i]
        state["photo1"] = _to_photo(img1)
        state["photo2"] = _to_photo(img2)
        label1.configure(image=state["photo1"])
        label2.configure(image=state["photo2"])

        r = float(np.asarray(state["rewards"][i]).squeeze())
        act = np.asarray(state["actions"][i]).squeeze()
        total_actions = int(np.asarray(state["actions"]).shape[0])
        status.configure(
            text=(
                f"demo {state['demo']}  step {i+1}/{cam_link0.shape[0]}  "
                f"actions={total_actions}  reward={r:.4f}  actions={act}"
            )
        )

    def _next_frame():
        state["idx"] += 1
        _update()

    def _prev_frame():
        state["idx"] -= 1
        _update()

    def _on_slider(value):
        try:
            state["idx"] = int(float(value))
            _update()
        except Exception:
            return

    def _delete_demo():
        path = state.get("path")
        if path and path.exists():
            file_name = path.name
            demo_key = state.get("demo", "")
            if demo_key:
                with h5py.File(path, "a") as f:
                    if demo_key in f["data"]:
                        del f["data"][demo_key]
                        status.configure(text=f"Deleted {file_name}::{demo_key}")
            _refresh_demos(file_name)

    # UI
    frame = ttk.Frame(root, padding=8)
    frame.grid(row=0, column=0, sticky="nsew")

    topbar = ttk.Frame(frame)
    topbar.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 6))
    ttk.Label(topbar, text="Dataset root:").pack(side=tk.LEFT, padx=(0, 6))
    root_entry = ttk.Entry(topbar, textvariable=root_var, width=60)
    root_entry.pack(side=tk.LEFT, padx=(0, 6))

    def _choose_root():
        selected = filedialog.askdirectory(initialdir=root_var.get())
        if selected:
            root_var.set(selected)
            _refresh_files()

    ttk.Button(topbar, text="Browse", command=_choose_root).pack(side=tk.LEFT, padx=4)
    ttk.Button(topbar, text="Refresh", command=lambda: _refresh_files()).pack(side=tk.LEFT, padx=6)
    ttk.Button(topbar, text="Quit", command=root.destroy).pack(side=tk.LEFT, padx=6)

    list_frame = ttk.Frame(frame)
    list_frame.grid(row=1, column=0, sticky="ns", padx=(0, 8))
    ttk.Label(list_frame, text="Files").pack(anchor="w")
    file_list = tk.Listbox(list_frame, height=8, width=28)
    file_list.pack(fill=tk.BOTH, expand=True)
    ttk.Label(list_frame, text="Demos").pack(anchor="w", pady=(6, 0))
    demo_list = tk.Listbox(list_frame, height=10, width=28)
    demo_list.pack(fill=tk.BOTH, expand=True)

    def _on_file_select(event):
        sel = file_list.curselection()
        if sel:
            idx = sel[0]
            val = file_list.get(idx)
            file_var.set(val)
            _refresh_demos(val)

    def _on_demo_select(event):
        sel = demo_list.curselection()
        if sel:
            idx = sel[0]
            val = demo_list.get(idx)
            demo_var.set(val)
            try:
                _load(file_var.get(), val)
            except Exception:
                return

    file_list.bind("<<ListboxSelect>>", _on_file_select)
    demo_list.bind("<<ListboxSelect>>", _on_demo_select)

    label1 = ttk.Label(frame)
    label2 = ttk.Label(frame)
    label1.grid(row=1, column=1, padx=5, pady=5)
    label2.grid(row=1, column=2, padx=5, pady=5)

    controls = ttk.Frame(frame)
    controls.grid(row=2, column=0, columnspan=3, pady=6)

    ttk.Button(controls, text="Prev Frame", command=_prev_frame).grid(row=0, column=0, padx=4)
    ttk.Button(controls, text="Next Frame", command=_next_frame).grid(row=0, column=1, padx=4)
    ttk.Button(controls, text="Delete Demo", command=_delete_demo).grid(row=0, column=2, padx=4)

    frame_slider = ttk.Scale(frame, from_=0, to=0, orient=tk.HORIZONTAL, command=_on_slider)
    frame_slider.grid(row=3, column=0, columnspan=3, sticky="ew", padx=6, pady=(0, 6))

    status = ttk.Label(frame, text="")
    status.grid(row=4, column=0, columnspan=3, pady=4)

    root.rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)

    _refresh_files(select_file=None)

    root.mainloop()


if __name__ == "__main__":
    main()
