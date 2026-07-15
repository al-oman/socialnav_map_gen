"""Extract per-floor occupancy maps for a scene.

Floor heights come from the episode file: scenes are multi-storey and every
episode carries its floor in start_position[1] — there is no consistent
global slice height across scenes (the old version picked a random navigable
point, i.e. a random floor).

The navmesh is computed with a near-zero agent radius, so the saved map is
the TRUE walkable geometry. Downstream agents add their own clearance
(robot margin, ORCA radii) — do not inflate the map and the agents both.

Assumes the episode file is per-scene (content/<scene>.json.gz style).
"""

import gzip
import json

import habitat_sim
import numpy as np


def floor_heights(episode_files, tol=0.5):
    """Distinct floor heights (m) among the episodes' start positions, across
    one or more episode files (e.g. train + val splits of the same scene).
    Heights within `tol` of each other are the same floor."""
    if isinstance(episode_files, str):
        episode_files = [episode_files]
    heights = []
    for path in episode_files:
        with gzip.open(path, "rt") as f:
            heights += [e["start_position"][1] for e in json.load(f)["episodes"]]
    heights = sorted(heights)
    floors = [[heights[0]]]
    for h in heights[1:]:
        if h - floors[-1][-1] > tol:
            floors.append([])
        floors[-1].append(h)
    return [(float(np.mean(f)), len(f)) for f in floors]  # (height, n_episodes)


# Vertical clearance per agent profile: where a person can walk vs where a
# Spot-sized robot can go (under tables, counters). One map set per profile.
CLEARANCE = {"human": 1.5, "spot": 0.4}


def open_scene(scene_id, scene_dataset_config=None):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_id
    if scene_dataset_config:
        sim_cfg.scene_dataset_config_file = scene_dataset_config
    return habitat_sim.Simulator(
        habitat_sim.Configuration(sim_cfg, [habitat_sim.agent.AgentConfiguration()])
    )


def set_clearance(sim, agent_height, agent_radius=0.05):
    """Recompute the navmesh for one agent profile. agent_radius ~0 keeps the
    walkable area un-eroded (one voxel) — agents add their own radius later."""
    ns = habitat_sim.NavMeshSettings()
    ns.set_defaults()
    ns.agent_radius = agent_radius
    ns.agent_height = agent_height
    sim.recompute_navmesh(sim.pathfinder, ns)


def extract_nav_map(sim, height, mpp=0.05):
    """Occupancy at one floor height. True = navigable.
    world_x = origin_x + col * mpp ; world_z = origin_z + row * mpp."""
    nav = sim.pathfinder.get_topdown_view(mpp, height)
    lower, _ = sim.pathfinder.get_bounds()
    return nav, {"mpp": mpp, "origin_x": lower[0], "origin_z": lower[2],
                 "height": height}


if __name__ == "__main__":
    scene = "/speed-scratch/al_oman/VLA/matterport/data/scene_datasets/mp3d/D7N2EKCX4Sj/D7N2EKCX4Sj.glb"
    dataset_cfg = "/speed-scratch/al_oman/VLA/matterport/data/scene_datasets/mp3d/mp3d.scene_dataset_config.json"
    episode_file = "data/D7N2EKCX4Sj.json.gz"  # per-scene episodes

    sim = open_scene(scene, dataset_cfg)
    floors = floor_heights(episode_file)
    for profile, clearance in CLEARANCE.items():
        set_clearance(sim, clearance)
        for i, (height, n_eps) in enumerate(floors):
            nav, meta = extract_nav_map(sim, height)
            out = f"scene_map_{profile}_floor{i}.npz"
            np.savez(out, nav=nav, clearance=clearance, **meta)
            print(f"{profile} floor {i}: height {height:.2f} m, {n_eps} episodes, "
                  f"{nav.shape} cells, {nav.mean():.0%} navigable -> {out}")
    sim.close()
