"""A live window onto a training run's instances.

    python -m train.watch --run runs/emerald

Shows the grid of screens the trainer writes to <run>/grid.png after every
step, and the latest progress line from <run>/log.csv. It only reads files, so
it can be opened and closed at any time without touching the run.
"""

import argparse
import csv
import os
import tkinter as tk


def latest_progress(run: str) -> str:
    try:
        with open(os.path.join(run, "log.csv"), newline="") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return "waiting for the first update"
    if not rows:
        return "waiting for the first update"
    r = rows[-1]
    return (f"update {r['update']}  step {int(r['global_step']):,}  {r['sps']} sps  |  "
            f"episode step {r['episode_step']}: tiles {float(r['tiles']):.1f}  "
            f"maps {float(r['maps']):.2f} (max {r['max_maps']})  party {float(r['party']):.2f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="runs/emerald")
    p.add_argument("--scale", type=float, default=1.0,
                   help="2 doubles the grid, 0.5 halves it (whole ratios only)")
    args = p.parse_args()
    grid_path = os.path.join(args.run, "grid.png")

    root = tk.Tk()
    root.title(f"gba-emulator-rl: {args.run}")
    root.configure(bg="#111")
    status = tk.Label(root, text="waiting for grid.png", fg="#ddd", bg="#111", font=("Consolas", 10), anchor="w")
    status.pack(fill="x", padx=6, pady=4)
    screen = tk.Label(root, bg="#111")
    screen.pack(padx=6, pady=(0, 6))

    state = {"mtime": 0.0, "image": None}

    def refresh():
        try:
            mtime = os.path.getmtime(grid_path)
            if mtime != state["mtime"]:
                image = tk.PhotoImage(file=grid_path)
                # PhotoImage scales by integer ratios only: zoom up, subsample down.
                if args.scale >= 1:
                    image = image.zoom(max(1, round(args.scale)))
                elif args.scale > 0:
                    image = image.subsample(max(1, round(1 / args.scale)))
                state["image"], state["mtime"] = image, mtime
                screen.configure(image=image)
        except (OSError, tk.TclError):
            pass  # not written yet, or replaced mid-read; try again shortly
        status.configure(text=latest_progress(args.run))
        root.after(500, refresh)

    refresh()
    root.mainloop()


if __name__ == "__main__":
    main()
