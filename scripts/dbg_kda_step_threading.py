"""debug E1.5：fused_recurrent_kda 整段 vs 逐步 threading 是否一致。
试 transpose_state_layout 等，定位 state 重喂约定。"""
import torch
from fla.ops.kda import fused_recurrent_kda
from fla.ops.kda.gate import fused_kda_gate

dev = "cuda"; N, T, H, Dh = 2, 8, 4, 64
torch.manual_seed(0)
q = torch.randn(N, T, H, Dh, device=dev, dtype=torch.bfloat16)
k = torch.randn(N, T, H, Dh, device=dev, dtype=torch.bfloat16)
v = torch.randn(N, T, H, Dh, device=dev, dtype=torch.bfloat16)
g_raw = torch.randn(N, T, H, Dh, device=dev)
A_log = torch.log(torch.empty(H, device=dev).uniform_(1, 16)); dt_bias = torch.zeros(H * Dh, device=dev)
g = fused_kda_gate(g_raw, A_log, dt_bias=dt_bias)
beta = torch.rand(N, T, H, device=dev)


def run(transpose):
    o_full, sf = fused_recurrent_kda(q=q, k=k, v=v, g=g, beta=beta, initial_state=None,
                                     output_final_state=True, use_qk_l2norm_in_kernel=True,
                                     transpose_state_layout=transpose)
    st = None; outs = []
    for t in range(T):
        ot, st = fused_recurrent_kda(q=q[:, t:t+1], k=k[:, t:t+1], v=v[:, t:t+1], g=g[:, t:t+1],
                                     beta=beta[:, t:t+1], initial_state=st, output_final_state=True,
                                     use_qk_l2norm_in_kernel=True, transpose_state_layout=transpose)
        outs.append(ot)
    o_step = torch.cat(outs, dim=1)
    return (o_full.float() - o_step.float()).abs().max().item(), \
           (sf.float() - st.float()).abs().max().item()


for tr in (False, True):
    od, sd = run(tr)
    print(f"transpose_state_layout={tr}: out maxdiff={od:.4e} | final_state maxdiff={sd:.4e}")
print("DBG_DONE")
