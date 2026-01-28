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


def _load_demo(path: Path):
    with h5py.File(path, "r") as f:
        demo_key = next(iter(f["data"].keys()))
        demo = f["data"][demo_key]
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
    color_var = tk.StringVar(value="all")

    def _demo_path(demo_key: str) -> Path:
        return Path(root_var.get()) / f"{demo_key}.hdf5"

    def _list_demos() -> list[str]:
        demos: list[str] = []
        for p in sorted(Path(root_var.get()).glob("demo_*.hdf5")):
            demos.append(p.stem)
        return demos

    def _load(demo_key: str) -> bool:
        path = _demo_path(demo_key)
        if not path.exists():
            return False
        cam_link0, cam_fixed, rewards, actions = _load_demo(path)
        state.update(
            {
                "demo": demo_key,
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

    def _find_next(start: str, direction: int) -> str | None:
        demos = _list_demos()
        if not demos:
            return None
        demos = sorted(demos)
        if start not in demos:
            return demos[0]
        idx = demos.index(start)
        next_idx = (idx + direction) % len(demos)
        return demos[next_idx]

    def _refresh_list(select_demo: str | None = None):
        demo_list.delete(0, tk.END)
        demos = _list_demos()
        for d in demos:
            demo_list.insert(tk.END, d)
        if demos:
            target = select_demo if select_demo is not None else demos[0]
            if target in demos:
                idx = demos.index(target)
                demo_list.selection_set(idx)
                demo_list.activate(idx)
                _load(target)
        else:
            state.update(
                {
                    "demo": 0,
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
            status.configure(text="No demos found for this color.")

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

    def _next_demo():
        nxt = _find_next(state["demo"], 1)
        if nxt is not None:
            _refresh_list(select_demo=nxt)

    def _prev_demo():
        prv = _find_next(state["demo"], -1)
        if prv is not None:
            _refresh_list(select_demo=prv)

    def _delete_demo():
        path = state.get("path")
        if path and path.exists():
            current = state.get("demo", "")
            demos_before = _list_demos()
            os.remove(path)
            status.configure(text=f"Deleted {path}")
            if current in demos_before:
                idx = demos_before.index(current)
                if demos_before:
                    nxt_idx = idx if idx < len(demos_before) - 1 else max(0, len(demos_before) - 2)
                else:
                    nxt_idx = 0
                demos_after = _list_demos()
                if demos_after:
                    if nxt_idx >= len(demos_after):
                        nxt_idx = len(demos_after) - 1
                    nxt = demos_after[nxt_idx]
                else:
                    nxt = None
            else:
                nxt = _find_next(current, 1)
            _refresh_list(select_demo=nxt)

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
            _refresh_list()

    ttk.Button(topbar, text="Browse", command=_choose_root).pack(side=tk.LEFT, padx=4)
    ttk.Button(topbar, text="Refresh", command=lambda: _refresh_list()).pack(side=tk.LEFT, padx=6)
    ttk.Button(topbar, text="Quit", command=root.destroy).pack(side=tk.LEFT, padx=6)

    list_frame = ttk.Frame(frame)
    list_frame.grid(row=1, column=0, sticky="ns", padx=(0, 8))
    ttk.Label(list_frame, text="Demos").pack(anchor="w")
    demo_list = tk.Listbox(list_frame, height=18, width=24)
    demo_list.pack(fill=tk.BOTH, expand=True)

    def _on_select(event):
        sel = demo_list.curselection()
        if sel:
            idx = sel[0]
            val = demo_list.get(idx)
            try:
                _load(val)
            except Exception:
                return

    demo_list.bind("<<ListboxSelect>>", _on_select)

    label1 = ttk.Label(frame)
    label2 = ttk.Label(frame)
    label1.grid(row=1, column=1, padx=5, pady=5)
    label2.grid(row=1, column=2, padx=5, pady=5)

    controls = ttk.Frame(frame)
    controls.grid(row=2, column=0, columnspan=3, pady=6)

    ttk.Button(controls, text="Prev Frame", command=_prev_frame).grid(row=0, column=0, padx=4)
    ttk.Button(controls, text="Next Frame", command=_next_frame).grid(row=0, column=1, padx=4)
    ttk.Button(controls, text="Prev Demo", command=_prev_demo).grid(row=0, column=2, padx=4)
    ttk.Button(controls, text="Next Demo", command=_next_demo).grid(row=0, column=3, padx=4)
    ttk.Button(controls, text="Delete Demo", command=_delete_demo).grid(row=0, column=4, padx=4)

    frame_slider = ttk.Scale(frame, from_=0, to=0, orient=tk.HORIZONTAL, command=_on_slider)
    frame_slider.grid(row=3, column=0, columnspan=3, sticky="ew", padx=6, pady=(0, 6))

    status = ttk.Label(frame, text="")
    status.grid(row=4, column=0, columnspan=3, pady=4)

    root.rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)

    _refresh_list(select_demo=args.demo or None)

    root.mainloop()


if __name__ == "__main__":
    main()
