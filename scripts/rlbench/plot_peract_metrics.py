"""Plot PerAct train and held-out validation metrics from TensorBoard events."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", required=True)
    args = parser.parse_args()

    event_files = sorted(args.run_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard events in {args.run_dir}")
    series = {}
    for path in event_files:
        events = EventAccumulator(str(path))
        events.Reload()
        for tag in events.Tags()["scalars"]:
            by_step = series.setdefault(tag, {})
            for point in events.Scalars(tag):
                by_step[point.step + 1] = point.value

    plots = [
        ("Position error (m)", "traj_pos_l2"),
        ("Position within 1 cm (%)", "traj_pos_acc_001"),
        ("Quaternion L1 error", "traj_rot_l1"),
        ("Quaternion accuracy (%)", "traj_rot_acc_0025"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, (title, metric) in zip(axes.flat, plots):
        for split in ("train", "val"):
            tag = f"{split}-losses/mean/{metric}"
            points = sorted(series.get(tag, {}).items())
            if not points:
                raise ValueError(f"Missing metric {tag} in {args.run_dir}")
            scale = 100 if "%" in title else 1
            ax.plot(
                [step for step, _ in points],
                [value * scale for _, value in points],
                marker="o", label=split,
            )
        ax.set_title(title)
        ax.set_xlabel("Training iteration")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle(f"PerAct {args.task}: held-out validation metrics", fontsize=16)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
