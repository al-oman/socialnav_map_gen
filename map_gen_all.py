"""Batch map extraction: every scene, every floor, both clearance profiles.

Reuses map_gen.py's fixed logic (floor heights from episode files, un-eroded
radius). Scenes without an episode file are skipped — no episodes means no
experiments there, so no map needed.

Run: python map_gen_all.py
"""

import glob
import os

import numpy as np

os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

from map_gen import CLEARANCE, extract_nav_map, floor_heights, open_scene, set_clearance

OUT = "/speed-scratch/al_oman/diffusion/socialnav_map_gen"
SCENE_ROOT = "/speed-scratch/al_oman/VLA/matterport/data/scene_datasets"
EPISODES_ROOT = "/speed-scratch/al_oman/VLA/matterport/data/datasets/pointnav"  # <- set to your json.gz root


def subdir_for(path):  # map a scene path -> output subfolder
    if "/mp3d/" in path:
        return "mp3d"
    if "/hm3d/train/" in path:
        return "hm3d/train"
    if "/hm3d/minival/" in path:
        return "hm3d/minival"
    if "/hm3d/val/" in path:
        return "hm3d/val"
    return "other"  # anything unexpected lands here, not lost


scenes = sorted(glob.glob(f"{SCENE_ROOT}/mp3d/**/*.glb", recursive=True) +
                glob.glob(f"{SCENE_ROOT}/hm3d/**/*.glb", recursive=True))
print(f"{len(scenes)} scene files")

for p in scenes:
    name = os.path.basename(p).split(".")[0]
    episode_files = glob.glob(f"{EPISODES_ROOT}/**/*{name}*.json.gz", recursive=True)
    if not episode_files:
        continue  # no episodes -> no experiments -> no map needed
    subdir = os.path.join(OUT, subdir_for(p))
    os.makedirs(subdir, exist_ok=True)
    if glob.glob(os.path.join(subdir, f"{name}_*_floor0.npz")):
        print("skip", name)
        continue
    try:
        floors = floor_heights(episode_files)  # merged across splits
        sim = open_scene(p)
        for profile, clearance in CLEARANCE.items():
            set_clearance(sim, clearance)
            for i, (height, n_eps) in enumerate(floors):
                nav, meta = extract_nav_map(sim, height)
                out = os.path.join(subdir, f"{name}_{profile}_floor{i}.npz")
                np.savez(out, nav=nav, clearance=clearance, **meta)
        sim.close()
        print(f"ok {name}: {len(floors)} floor(s) x {len(CLEARANCE)} profiles")
    except Exception as e:
        print("FAIL", name, repr(e))
