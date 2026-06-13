"""Phase-1 (habitat025): roll out short nav trajectories in HM3D val scenes and render RGB frame
sequences for nav post-training of the linearized Causal-Forcing WM.

For each clip:
  * pick a random navigable start point + random yaw (matches the eval sensor: 512x512, hfov 90,
    sensor_position 1.0, agent_height 1.0, agent_radius 0.1, pitch 0  -- from downstream/simulator.py
    default_settings, so the conditioning first-frame distribution matches IGNav eval).
  * sample a length-`motions` action sequence over {1:forward, 2:turn_left, 3:turn_right}
    (UNIT_DISTANCE=0.20 m, UNIT_DEGREE=22.5deg -- the exact eval primitives).
  * execute each primitive as `substeps` micro-steps (forward 0.20/substeps m; turn 22.5/substeps deg),
    rendering one RGB frame after EACH micro-step -> smooth video. 1 initial + motions*substeps frames.
    With motions=4, substeps=4 -> 17 px frames -> Wan VAE (4x temporal) -> 5 latent frames, matching
    eval's num_latent_frames=5.

Saves per clip a dir frames/clip_XXXXX/ with frame_00.png..frame_NN.png + meta.json holding
  b_action = [0] + motion_ids   (the placeholder-prefixed id list process_b_action_nav consumes;
                                 it drops [0] and renders the eval text).

Run (habitat025):
  MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet \
  /root/miniconda3/envs/habitat025/bin/python navgen_render.py --scenes 5 --per_scene 12 \
      --out /root/nav_posttrain/frames
"""
import argparse, gzip, json, os, random, sys
import numpy as np
import imageio.v2 as iio
import habitat_sim

REPO = "/home/wm-vla-rl/_ext/world-in-world"
HM3D_VAL = os.path.join(REPO, "data/scene_datasets/hm3d/val")
SCENE_CFG = os.path.join(REPO, "data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json")

UNIT_DISTANCE = 0.20   # downstream/prompts.py
UNIT_DEGREE = 22.5     # downstream/prompts.py
# eval sensor (downstream/simulator.py default_settings)
SENSOR_W, SENSOR_H, HFOV, SENSOR_POS, AGENT_H, AGENT_R = 512, 512, 90.0, 1.0, 1.0, 0.1

MOTION_IDS = [1, 2, 3]  # forward, turn_left, turn_right (IGENEX_ACTION_IDS, excl stop/placeholder)


def make_sim(glb, substeps, device_id=0):
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = glb
    backend.scene_dataset_config_file = SCENE_CFG
    backend.enable_physics = False
    backend.gpu_device_id = device_id
    backend.random_seed = 1

    rgb = habitat_sim.CameraSensorSpec()
    rgb.uuid = "rgb_sensor"
    rgb.sensor_type = habitat_sim.SensorType.COLOR
    rgb.resolution = [SENSOR_H, SENSOR_W]
    rgb.position = [0.0, SENSOR_POS, 0.0]
    rgb.hfov = HFOV

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.height = AGENT_H
    agent_cfg.radius = AGENT_R
    agent_cfg.sensor_specifications = [rgb]
    # micro-step actions: each primitive is `substeps` of these
    fwd_amt = UNIT_DISTANCE / substeps
    turn_amt = UNIT_DEGREE / substeps
    A = habitat_sim.agent.ActionSpec
    Act = habitat_sim.agent.ActuationSpec
    agent_cfg.action_space = {
        "fwd_micro": A("move_forward", Act(amount=fwd_amt)),
        "left_micro": A("turn_left", Act(amount=turn_amt)),
        "right_micro": A("turn_right", Act(amount=turn_amt)),
    }
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent_cfg]))
    return sim


MICRO = {1: "fwd_micro", 2: "left_micro", 3: "right_micro"}


def rollout(sim, agent, motions, substeps, rng):
    """Place at a random navigable start + random yaw, execute the motion sequence as micro-steps,
    return (frames list[HxWx3 uint8], displacement)."""
    pf = sim.pathfinder
    start = pf.get_random_navigable_point()
    yaw = rng.uniform(0, 2 * np.pi)
    st = agent.get_state()
    st.position = np.array(start, dtype=np.float32)
    st.rotation = np.quaternion(np.cos(yaw / 2), 0, np.sin(yaw / 2), 0)  # yaw about +Y
    agent.set_state(st)
    p0 = np.array(agent.get_state().position, dtype=np.float32)

    frames = [sim.get_sensor_observations()["rgb_sensor"][:, :, :3].copy()]  # initial
    for mid in motions:
        act = MICRO[mid]
        for _ in range(substeps):
            obs = sim.step(act)
            frames.append(obs["rgb_sensor"][:, :, :3].copy())
    p1 = np.array(agent.get_state().position, dtype=np.float32)
    disp = float(np.linalg.norm(p1 - p0))
    return frames, disp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/root/nav_posttrain/frames")
    ap.add_argument("--scenes", type=int, default=5, help="number of HM3D val scenes to use")
    ap.add_argument("--per_scene", type=int, default=12, help="clips per scene")
    ap.add_argument("--motions", type=int, default=4, help="action-sequence length (eval look-ahead)")
    ap.add_argument("--substeps", type=int, default=4, help="micro-steps per primitive (4 -> 17 frames)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device_id", type=int, default=0)
    ap.add_argument("--min_disp", type=float, default=0.0,
                    help="skip clips whose net displacement < this (avoid all-static clips)")
    ap.add_argument("--max_static_frac", type=float, default=0.6,
                    help="skip clip if > this fraction of consecutive frames are ~identical (stuck on wall)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    nrng = np.random.RandomState(args.seed)

    scenes = sorted(d for d in os.listdir(HM3D_VAL) if os.path.isdir(os.path.join(HM3D_VAL, d)))
    # prefer the val-split scene-id range (>=00803) that the IGNav episodes reference
    scenes = [s for s in scenes if s.split("-")[0] >= "00803"] or scenes
    rng.shuffle(scenes)
    scenes = scenes[: args.scenes]
    print(f"[render] {len(scenes)} scenes x {args.per_scene} clips, motions={args.motions} "
          f"substeps={args.substeps} -> {1 + args.motions * args.substeps} frames/clip", flush=True)

    clip_id = len([d for d in os.listdir(args.out) if d.startswith("clip_")])
    n_made = 0
    for sc in scenes:
        glb = os.path.join(HM3D_VAL, sc, sc.split("-")[1] + ".basis.glb")
        if not os.path.exists(glb):
            print(f"[render] skip {sc}: no glb", flush=True); continue
        try:
            sim = make_sim(glb, args.substeps, args.device_id)
        except Exception as e:
            print(f"[render] skip {sc}: sim init failed {e}", flush=True); continue
        if not sim.pathfinder.is_loaded:
            print(f"[render] skip {sc}: no navmesh", flush=True); sim.close(); continue
        agent = sim.get_agent(0)
        made_here = 0
        attempts = 0
        while made_here < args.per_scene and attempts < args.per_scene * 4:
            attempts += 1
            motions = [rng.choice(MOTION_IDS) for _ in range(args.motions)]
            try:
                frames, disp = rollout(sim, agent, motions, args.substeps, nrng)
            except Exception as e:
                print(f"[render] {sc} rollout err: {e}", flush=True); continue
            if disp < args.min_disp:
                continue
            # static check: fraction of near-identical consecutive frames
            diffs = [np.mean(np.abs(frames[i].astype(np.int16) - frames[i - 1].astype(np.int16)))
                     for i in range(1, len(frames))]
            static_frac = np.mean([d < 1.0 for d in diffs])
            if static_frac > args.max_static_frac:
                continue
            cdir = os.path.join(args.out, f"clip_{clip_id:05d}")
            os.makedirs(cdir, exist_ok=True)
            for i, fr in enumerate(frames):
                iio.imwrite(os.path.join(cdir, f"frame_{i:02d}.png"), fr)
            meta = {
                "scene": sc, "b_action": [0] + motions, "motions": motions,
                "n_frames": len(frames), "substeps": args.substeps,
                "disp": disp, "mean_frame_diff": float(np.mean(diffs)),
            }
            with open(os.path.join(cdir, "meta.json"), "w") as f:
                json.dump(meta, f)
            clip_id += 1; made_here += 1; n_made += 1
            if n_made % 10 == 0:
                print(f"[render] {n_made} clips (scene {sc}, disp {disp:.2f}, "
                      f"frame_diff {np.mean(diffs):.1f})", flush=True)
        print(f"[render] scene {sc}: {made_here} clips ({attempts} attempts)", flush=True)
        sim.close()
    print(f"=== [render] DONE: {n_made} new clips in {args.out} (total {clip_id}) ===", flush=True)


if __name__ == "__main__":
    main()
