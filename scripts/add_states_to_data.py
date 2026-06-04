import os, sys, glob, h5py, numpy as np, torch
sys.path.insert(0, "/home/wm-vla-rl/_ext/LIBERO")
STRIDE = 2; PER_TASK = 10
data = torch.load("/home/wm-vla-rl/_ext/datasets/latentvla_align_data.pt", map_location="cpu")
idx = 0
for suite in ["libero_90", "libero_10"]:
    files = sorted(glob.glob(f"/home/wm-vla-rl/_ext/datasets/LIBERO/{suite}/*.hdf5"))
    for f in files:
        with h5py.File(f, "r") as hf:
            for k in list(hf["data"].keys())[:PER_TASK]:
                d = hf["data"][k]
                st = np.concatenate([d["obs"]["ee_pos"][::STRIDE], d["obs"]["ee_ori"][::STRIDE],
                                     d["obs"]["gripper_states"][::STRIDE][:, :1]], -1).astype(np.float32)
                n = data[idx]["latent"].shape[0]
                data[idx]["states"] = torch.tensor(st[:n])
                assert data[idx]["T"] == n
                idx += 1
torch.save(data, "/home/wm-vla-rl/_ext/datasets/latentvla_align_data.pt")
print(f"ADD_STATES_DONE updated {idx} demos with proprio; example states shape {data[0]['states'].shape}")
