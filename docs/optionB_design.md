# Option B:重对齐 OpenVLA-OFT 视觉前端到 V-JEPA2(2026-06-04)
**思路**:保住 OFT 已训好的 LLM(7B)+ action head(competent,94.5%),**只换视觉前端**(SigLIP/DINO → V-JEPA2 + 新 projector),轻量重对齐。比从零造 LatentVLA(线2,0%)强得多——用 OFT 的容量。

## 注入点(已读 OFT 代码确认)
`PrismaticForActionPrediction._process_vision_features(pixel_values)`:
  pixel_values → vision_backbone(SigLIP+DINO)→ projector → **projected_patch_embeddings [bsz, 256, llm_dim]** → 进 LLM。
**override 它**:V-JEPA2-latent[bsz, **256**, 1408] → 新 projector → [bsz, 256, llm_dim]。**V-JEPA2 原生 256 token = OFT 256,token 数天然匹配**。其余(LLM/action head/proprio/get_action)全复用 OFT。

## 实现(monkeypatch,最小手术)
1. `initialize_model` 载 OFT(model + action_head + proprio_projector + processor)。
2. 加 `model.vjepa_projector`(1408→llm_dim MLP)。
3. monkeypatch `model._process_vision_features = lambda pv: model.vjepa_projector(pv)`(pv = V-JEPA2 latent 当 pixel_values 传)。
4. **重对齐训练**:LIBERO demo,encode obs→V-JEPA2 latent→当 pixel_values→OFT forward→L1 loss;训 **vjepa_projector + LLM LoRA**,action head 冻结(保 OFT 能力)。
5. 闭环 eval:get_action(monkeypatch 生效)→ sim success,对标 OFT 原版 94.5%。

## 验证(已通过)
- V-JEPA2 + OFT 同 oft venv 加载 ✅(protobuf python impl)。
- 注入点 = `_process_vision_features`,256 token 匹配 ✅。

## 判据
重对齐后闭环 success vs OFT 原版(像素)94.5%。若接近(如 >50%)→ V-JEPA2 latent 能承载 OFT 所需信息、重对齐成功 → 可进 imagination RL。诚实风险:JEPA 特征若不含 OFT 依赖的细节,会掉点。
