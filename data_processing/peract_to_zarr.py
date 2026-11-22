import argparse
import json
import os
from pathlib import Path
import pickle

import blosc
from numcodecs import Blosc
import zarr
import numpy as np
import torch
from tqdm import tqdm

STORE_EVERY = 1
NCAM = 4
NHAND = 1
IM_SIZE = 256

PERACT_TASKS = [
    "place_cups", "close_jar", "insert_onto_square_peg",
    "light_bulb_in", "meat_off_grill", "open_drawer",
    "place_shape_in_shape_sorter", "place_wine_at_rack_location",
    "push_buttons", "put_groceries_in_cupboard",
    "put_item_in_drawer", "put_money_in_safe", "reach_and_drag",
    "slide_block_to_color_target", "stack_blocks", "stack_cups",
    "sweep_to_dustpan_of_size", "turn_tap"
]


def parse_arguments():
    parser = argparse.ArgumentParser()
    # Tuples: (name, type, default)
    arguments = [
        ('root', str, '/data/group_data/katefgroup/VLA/Peract_packaged/'),
        ('tgt', str, '/data/user_data/ngkanats/zarr_datasets/Peract_dat_zarr/'),
        ('instructions', str, 'instructions/peract/instructions.json')
    ]
    for arg in arguments:
        parser.add_argument(f'--{arg[0]}', type=arg[1], default=arg[2])
    parser.add_argument(
        '--tasks',
        nargs='+',
        choices=PERACT_TASKS,
        default=['push_buttons', 'stack_cups'],
        help='PerAct tasks to package (default: push_buttons stack_cups)'
    )

    return parser.parse_args()


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        return np.array(x)


def all_tasks_main(split, tasks):
    # Check if the zarr already exists
    filename = f"{STORE_PATH}/{split}.zarr"
    if os.path.exists(filename):
        print(f"Zarr file {filename} already exists. Skipping...")
        return None

    cameras = ["left_shoulder", "right_shoulder", "wrist", "front"]
    # Keep IDs compatible with datasets.rlbench.PERACT_TASKS even when only a
    # subset of tasks is converted.
    task2id = {task: PERACT_TASKS.index(task) for task in tasks}

    # Initialize zarr
    compressor = Blosc(cname='lz4', clevel=1, shuffle=Blosc.SHUFFLE)
    with zarr.open_group(filename, mode="w") as zarr_file:

        def _create(field, shape, dtype):
            zarr_file.create_dataset(
                field,
                shape=(0,) + shape,
                chunks=(STORE_EVERY,) + shape,
                compressor=compressor,
                dtype=dtype
            )

        _create("rgb", (NCAM, 3, IM_SIZE, IM_SIZE), "uint8")
        _create("pcd", (NCAM, 3, IM_SIZE, IM_SIZE), "float16")
        _create("proprioception", (3, NHAND, 8), "float32")
        _create("action", (1, NHAND, 8), "float32")
        _create("task_id", (), "uint8")
        _create("variation", (), "uint8")

        # Loop through episodes
        for task in tasks:
            print(task)
            episodes = []
            for var in range(0, 199):
                _path = Path(f'{ROOT}{split}/{task}+{var}/')
                if not _path.is_dir():
                    continue
                episodes.extend([
                    (ep, var) for ep in sorted(_path.glob("*.dat"))
                ])
            for ep, var in tqdm(episodes):
                # Read
                with open(ep, "rb") as f:
                    content = pickle.loads(blosc.decompress(f.read()))
                # Map [-1, 1] to [0, 255] uint8
                rgb = (127.5 * (content[1][:, :, 0] + 1)).astype(np.uint8)
                # Keep point cloud as it's hard to reverse
                pcd = content[1][:, :, 1].astype(np.float16)
                # Store current eef pose as well as two previous ones
                prop = np.stack([
                    to_numpy(tens).astype(np.float32) for tens in content[4]
                ])
                prop_1 = np.concatenate([prop[:1], prop[:-1]])
                prop_2 = np.concatenate([prop_1[:1], prop_1[:-1]])
                prop = np.concatenate([prop_2, prop_1, prop], 1)
                prop = prop.reshape(len(prop), 3, NHAND, 8)
                # Next keypose (concatenate curr eef to form a "trajectory")
                actions = np.stack([
                    to_numpy(tens).astype(np.float32) for tens in content[2]
                ]).reshape(len(content[2]), 1, NHAND, 8)
                # Task ids and variation ids
                tids = np.array([task2id[task]] * len(content[0])).astype(np.uint8)
                _vars = np.array([var] * len(content[0])).astype(np.uint8)

                # write
                zarr_file['rgb'].append(rgb)
                zarr_file['pcd'].append(pcd)
                zarr_file['proprioception'].append(prop)
                zarr_file['action'].append(actions)
                zarr_file['task_id'].append(tids)
                zarr_file['variation'].append(_vars)
                assert all(
                    len(zarr_file['action']) == len(zarr_file[key])
                    for key in zarr_file.keys()
                )


if __name__ == "__main__":
    args = parse_arguments()
    tasks = args.tasks
    ROOT = args.root
    STORE_PATH = args.tgt
    # Create zarr data
    for split in ['train', 'val']:
        all_tasks_main(split, tasks)
    # The packaged .dat tree does not contain RLBench's
    # variation_descriptions.pkl files. Reuse the instruction metadata shipped
    # with this repository and verify that it covers the selected tasks.
    with open(args.instructions) as fid:
        instr_dict = json.load(fid)
    missing_tasks = sorted(set(tasks) - set(instr_dict))
    if missing_tasks:
        raise ValueError(
            f"Missing tasks in {args.instructions}: {', '.join(missing_tasks)}"
        )
    print(f"Using instruction metadata from {args.instructions}")
