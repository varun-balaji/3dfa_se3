"""Record one held-out PerAct RLBench rollout as MP4 and GIF."""

import argparse
import json
import random
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from pyrep.errors import ConfigurationPathError, IKError
from rlbench.backend.exceptions import InvalidActionError

from online_evaluation_rlbench.evaluate_policy import load_models
from online_evaluation_rlbench.get_stored_demos import get_stored_demos
from online_evaluation_rlbench.utils_with_rlbench import (
    Actioner,
    Mover,
    RLBenchEnv,
    task_file_to_task_class,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("push_buttons", "stack_cups", "place_cups"), default="push_buttons")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("peract_raw/test"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variation", type=int, default=0)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--frame-stride", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = args.output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    for old_frame in frames_dir.glob("frame_*.png"):
        old_frame.unlink()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    model_args = SimpleNamespace(
        checkpoint=str(args.checkpoint), model_type="denoise3d", backbone="clip",
        num_vis_instr_attn_layers=2, fps_subsampling_factor=4,
        embedding_dim=120, num_attn_heads=8, num_history=3, bimanual=False,
        num_shared_attn_layers=4, relative_action=False,
        rotation_format="quat_xyzw", denoise_timesteps=5,
        denoise_model="rectified_flow",
    )
    actioner = Actioner(load_models(model_args), backbone="clip")
    env = RLBenchEnv(
        data_path=args.data_dir, image_size=(256, 256), apply_rgb=True,
        apply_pc=True, headless=True, apply_cameras=("wrist", "front"),
    )

    font = ImageFont.truetype("DejaVuSans.ttf", 14)
    frame_count = 0
    action_step = 0
    sim_step = 0
    success = False
    error = None

    def save_frame(front, wrist, status=""):
        nonlocal frame_count
        canvas = Image.new("RGB", (512, 282), "#181818")
        canvas.paste(Image.fromarray(front), (0, 26))
        canvas.paste(Image.fromarray(wrist), (256, 26))
        draw = ImageDraw.Draw(canvas)
        title = f"{args.task} v{args.variation} e{args.episode}  |  action {action_step}"
        draw.text((8, 5), title, fill="white", font=font)
        if status:
            draw.rectangle((365, 3, 507, 24), fill="#176d39" if status == "SUCCESS" else "#923333")
            draw.text((375, 5), status, fill="white", font=font)
        canvas.save(frames_dir / f"frame_{frame_count:05d}.png")
        frame_count += 1

    def capture_sensor(sensor):
        sensor.handle_explicitly()
        return np.clip(sensor.capture_rgb() * 255, 0, 255).astype(np.uint8)

    def capture_step():
        nonlocal sim_step
        sim_step += 1
        if sim_step % args.frame_stride == 0:
            save_frame(capture_sensor(scene._cam_front), capture_sensor(scene._cam_wrist))

    env.env.launch()
    try:
        task = env.env.get_task(task_file_to_task_class(args.task))
        task.set_variation(args.variation)
        demo = get_stored_demos(
            amount=1, dataset_root=str(args.data_dir),
            variation_number=args.variation, task_name=args.task,
            random_selection=False, from_episode_number=args.episode,
        )[0]
        descriptions, obs = task.reset_to_demo(demo)
        actioner.load_episode(descriptions)
        scene = task._scene
        save_frame(obs.front_rgb, obs.wrist_rgb)
        scene.register_step_callback(capture_step)
        mover = Mover(task, max_tries=2)
        grippers = torch.empty((1, 0, 8), device="cuda")

        with torch.inference_mode():
            for action_step in range(1, args.max_steps + 1):
                rgb, pcd, gripper = env.get_rgb_pcd_gripper_from_obs(obs)
                grippers = torch.cat((grippers, gripper.cuda().unsqueeze(1)), dim=1)
                history = grippers[:, -3:]
                history = F.pad(history, (0, 0, 3 - history.shape[1], 0), mode="replicate")
                actions = actioner.predict(rgb.cuda(), pcd.cuda(), history)[-1].cpu().numpy()
                actions[:, -1] = actions[:, -1].round()
                try:
                    for action in actions:
                        obs, reward, _ = mover(action, collision_checking=False)
                except (IKError, ConfigurationPathError, InvalidActionError) as exc:
                    error = str(exc)
                    break
                save_frame(obs.front_rgb, obs.wrist_rgb)
                if reward == 1:
                    success = True
                    break
        scene.register_step_callback(None)
        save_frame(
            obs.front_rgb, obs.wrist_rgb,
            status="SUCCESS" if success else "FAILURE",
        )
        final_frame = Image.open(frames_dir / f"frame_{frame_count - 1:05d}.png")
        for _ in range(9):
            final_frame.save(frames_dir / f"frame_{frame_count:05d}.png")
            frame_count += 1
    finally:
        env.env.shutdown()

    pattern = str(frames_dir / "frame_%05d.png")
    mp4 = args.output_dir / "rollout.mp4"
    gif = args.output_dir / "rollout.gif"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", "10", "-i", pattern,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(mp4),
    ], check=True)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", "10", "-i", pattern,
        "-vf", "fps=10", "-loop", "0", str(gif),
    ], check=True)
    metadata = {
        "task": args.task, "split": "test", "variation": args.variation,
        "episode": args.episode, "checkpoint": str(args.checkpoint),
        "actions": action_step, "frames": frame_count, "success": success,
        "error": error,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))
    print(mp4)
    print(gif)


if __name__ == "__main__":
    main()
