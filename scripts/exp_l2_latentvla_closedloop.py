"""(LatentVLA gate) 闭环 sim success —— 对齐质量的真 gate(研究:offline MSE 不预测闭环)。
V-JEPA2 编码器(池化 64-token)→ LatentVLA(projector+Qwen3+LoRA+proprio+action head)→ action chunk → step。
对比 full vs vision-zeroed(blind)策略:闭环 success 必须超 blind,才算编码器对齐真有用。
单 venv(latentvla:torch2.5+transformers4.53+vjepa2+libero)。
"""
import os, sys, json, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("HF_HOME", "/home/.hf_home")
os.environ.setdefault("TORCH_HOME", "/home/.cache/torch")
sys.path.insert(0, "/home/wm-vla-rl/_ext/vjepa2"); sys.path.insert(0, "/home/wm-vla-rl/_ext/LIBERO")
sys.path.insert(0, "/home/wm-vla-rl/module1_vjepa2ac"); sys.path.insert(0, "/home/wm-vla-rl/scripts")
from transformers import AutoModel, AutoTokenizer
from hubconf import vjepa2_ac_vit_giant
from app.vjepa_droid.transforms import make_transforms
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from scipy.spatial.transform import Rotation

DEV = "cuda"; CKPT = "/home/wm-vla-rl/_ext/ckpts"; RES = "/home/wm-vla-rl/module1_vjepa2ac/results"
LLM = os.environ.get("L2_LLM", "Qwen/Qwen3-1.7B"); CHUNK = int(os.environ.get("L2_CHUNK", "8"))
BENCH = os.environ.get("L2_BENCH", "libero_10"); NTASK = int(os.environ.get("L2_NTASK", "5"))
NEP = int(os.environ.get("L2_EP", "3")); HMAX = int(os.environ.get("L2_H", "300")); POOL = 8
BLIND = int(os.environ.get("L2_BLIND", "0"))            # 1=vision-zeroed(blind 对照)
WM_ENC = "kda_arlayer_big_sftfull.pt"


class LatentVLA(nn.Module):
    def __init__(self, llm, hid):
        super().__init__()
        self.llm = llm
        self.projector = nn.Sequential(nn.Linear(1408, hid), nn.GELU(), nn.Linear(hid, hid))
        self.proprio_proj = nn.Sequential(nn.Linear(7, hid), nn.GELU(), nn.Linear(hid, hid))
        self.action_head = nn.Sequential(nn.LayerNorm(hid), nn.Linear(hid, 512), nn.GELU(), nn.Linear(512, 7 * CHUNK))

    def forward(self, latent, proprio, le, lm):
        vis = self.projector(latent); pro = self.proprio_proj(proprio).unsqueeze(1)
        seq = torch.cat([vis, pro, le], dim=1)
        pre = torch.ones(vis.shape[0], vis.shape[1] + 1, device=vis.device, dtype=lm.dtype)
        out = self.llm(inputs_embeds=seq, attention_mask=torch.cat([pre, lm], 1)).last_hidden_state
        return self.action_head(out[:, -1]).view(-1, CHUNK, 7)


def main():
    enc, _ = vjepa2_ac_vit_giant(pretrained=True)
    enc.load_state_dict(torch.load(f"{CKPT}/{WM_ENC}", map_location="cpu")["encoder"])
    enc = enc.to(DEV).eval()
    tok = AutoTokenizer.from_pretrained(LLM)
    llm = AutoModel.from_pretrained(LLM, torch_dtype=torch.bfloat16).to(DEV)
    from peft import LoraConfig, get_peft_model
    llm = get_peft_model(llm, LoraConfig(r=32, lora_alpha=32, lora_dropout=0.05,
          target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    hid = llm.base_model.config.hidden_size
    model = LatentVLA(llm, hid).to(DEV)
    model.projector = model.projector.to(torch.bfloat16); model.proprio_proj = model.proprio_proj.to(torch.bfloat16); model.action_head = model.action_head.to(torch.bfloat16)
    sd = torch.load(f"{CKPT}/latentvla_align.pt", map_location="cpu")
    model.projector.load_state_dict(sd["projector"]); model.proprio_proj.load_state_dict(sd["proprio_proj"]) if "proprio_proj" in sd else None
    model.action_head.load_state_dict(sd["action_head"])
    if "lora" in sd: model.load_state_dict({k: v for k, v in sd["lora"].items()}, strict=False)
    model.eval()
    transform = make_transforms(crop_size=256, random_horizontal_flip=False, random_resize_aspect_ratio=(1., 1.),
                                random_resize_scale=(1., 1.), reprob=0., auto_augment=False, motion_shift=False)

    @torch.no_grad()
    def encode(img):
        clip = transform(img[None]).unsqueeze(0).to(DEV)
        B, C, T, H, W = clip.size()
        c = clip.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
        h = enc(c).view(1, -1, enc.embed_dim)
        h = F.layer_norm(h.float(), (h.size(-1),))
        g = int(h.shape[1] ** 0.5)
        h = h.view(1, g, g, h.shape[2]).permute(0, 3, 1, 2)
        h = F.adaptive_avg_pool2d(h, (POOL, POOL))
        return h.flatten(2).permute(0, 2, 1).to(torch.bfloat16)   # [1,64,1408]

    bench = benchmark.get_benchmark_dict()[BENCH]()
    n = min(NTASK, bench.n_tasks)
    print(f"L2 closed-loop | {BENCH} ntask={n} ep={NEP} blind={BLIND} chunk={CHUNK}", flush=True)
    succ = 0; tot = 0
    for ti in range(n):
        task = bench.get_task(ti)
        t = tok(task.language, return_tensors="pt").to(DEV)
        with torch.no_grad(): le = model.llm.base_model.get_input_embeddings()(t.input_ids).to(torch.bfloat16)
        lm = t.attention_mask
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256); env.seed(0)
        for ep in range(NEP):
            obs = env.reset(); ok = False
            for step in range(HMAX // CHUNK):
                z = encode(np.asarray(obs["agentview_image"]))
                if BLIND: z = torch.zeros_like(z)
                pos = obs["robot0_eef_pos"]; eul = Rotation.from_quat(obs["robot0_eef_quat"]).as_euler("xyz")
                grip = obs["robot0_gripper_qpos"][:1]
                pro = torch.tensor(np.concatenate([pos, eul, grip]), dtype=torch.bfloat16, device=DEV).view(1, 7)
                with torch.no_grad(): chunk = model(z, pro, le, lm)[0].float().cpu().numpy()
                done = False
                for a in chunk:
                    obs, _, done, _ = env.step(a.astype(np.float32))
                    if env.check_success(): ok = True; break
                    if done: break
                if ok or done: break
            succ += int(ok); tot += 1
            print(f"  task{ti} ep{ep}: {'SUCCESS' if ok else 'fail'}", flush=True)
        env.close()
    sr = succ / max(1, tot)
    json.dump(dict(bench=BENCH, blind=BLIND, success_rate=sr, n=tot), open(f"{RES}/latentvla_closedloop{'_blind' if BLIND else ''}.json", "w"))
    print(f"L2_DONE blind={BLIND} success={sr:.3f} ({succ}/{tot})", flush=True)


if __name__ == "__main__":
    main()
