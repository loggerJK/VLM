blip3oQwenForInferenceLM(
  (visual): Qwen2_5_VisionTransformerPretrainedModel( # Qwen2.5 ViT
    (patch_embed): Qwen2_5_VisionPatchEmbed(
      (proj): Conv3d(3, 1280, kernel_size=(2, 14, 14), stride=(2, 14, 14), bias=False)
    )
    (rotary_pos_emb): Qwen2_5_VisionRotaryEmbedding()
    (blocks): ModuleList(
      (0-31): 32 x Qwen2_5_VLVisionBlock(
        (norm1): Qwen2RMSNorm((1280,), eps=1e-06)
        (norm2): Qwen2RMSNorm((1280,), eps=1e-06)
        (attn): Qwen2_5_VLVisionSdpaAttention(
          (qkv): Linear(in_features=1280, out_features=3840, bias=True)
          (proj): Linear(in_features=1280, out_features=1280, bias=True)
        )
        (mlp): Qwen2_5_VLMLP(
          (gate_proj): Linear(in_features=1280, out_features=3420, bias=True)
          (up_proj): Linear(in_features=1280, out_features=3420, bias=True)
          (down_proj): Linear(in_features=3420, out_features=1280, bias=True)
          (act_fn): SiLU()
        )
      )
    )
    (merger): Qwen2_5_VLPatchMerger(
      (ln_q): Qwen2RMSNorm((1280,), eps=1e-06)
      (mlp): Sequential(
        (0): Linear(in_features=5120, out_features=5120, bias=True)
        (1): GELU(approximate='none')
        (2): Linear(in_features=5120, out_features=3584, bias=True)
      )
    )
  )
  (model): blip3oQwenModel(
    (embed_tokens): Embedding(151668, 3584, padding_idx=151643)
    (layers): ModuleList(
      (0-27): 28 x Qwen2_5_VLDecoderLayer(
        (self_attn): Qwen2_5_VLSdpaAttention(
          (q_proj): Linear(in_features=3584, out_features=3584, bias=True)
          (k_proj): Linear(in_features=3584, out_features=512, bias=True)
          (v_proj): Linear(in_features=3584, out_features=512, bias=True)
          (o_proj): Linear(in_features=3584, out_features=3584, bias=False)
          (rotary_emb): Qwen2_5_VLRotaryEmbedding()
        )
        (mlp): Qwen2MLP(
          (gate_proj): Linear(in_features=3584, out_features=18944, bias=False)
          (up_proj): Linear(in_features=3584, out_features=18944, bias=False)
          (down_proj): Linear(in_features=18944, out_features=3584, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): Qwen2RMSNorm((3584,), eps=1e-06)
        (post_attention_layernorm): Qwen2RMSNorm((3584,), eps=1e-06)
      )
    )
    (norm): Qwen2RMSNorm((3584,), eps=1e-06)
    (rotary_emb): Qwen2_5_VLRotaryEmbedding()
    (gen_vision_tower): EvaClipVisionTower()
    (dit): NextDiTCrossAttn(
      (model): LuminaNextDiT2DModel(
        (caption_projection): PixArtAlphaTextProjection(
          (linear_1): Linear(in_features=3584, out_features=1792, bias=True)
          (act_1): GELU(approximate='tanh')
          (linear_2): Linear(in_features=1792, out_features=1792, bias=True)
        )
        (patch_embedder): LuminaPatchEmbed(
          (proj): Linear(in_features=1792, out_features=1792, bias=True)
        )
        (time_caption_embed): LuminaCombinedTimestepCaptionEmbedding(
          (time_proj): Timesteps()
          (timestep_embedder): TimestepEmbedding(
            (linear_1): Linear(in_features=256, out_features=1024, bias=True)
            (act): SiLU()
            (linear_2): Linear(in_features=1024, out_features=1024, bias=True)
          )
          (caption_embedder): Sequential(
            (0): LayerNorm((1792,), eps=1e-05, elementwise_affine=True)
            (1): Linear(in_features=1792, out_features=1024, bias=True)
          )
        )
        (layers): ModuleList(
          (0-23): 24 x LuminaNextDiTBlock(
            (attn1): Attention(
              (norm_q): LayerNorm((1792,), eps=1e-05, elementwise_affine=True)
              (norm_k): LayerNorm((1792,), eps=1e-05, elementwise_affine=True)
              (to_q): Linear(in_features=1792, out_features=1792, bias=False)
              (to_k): Linear(in_features=1792, out_features=1792, bias=False)
              (to_v): Linear(in_features=1792, out_features=1792, bias=False)
              (to_out): Identity()
            )
            (attn2): Attention(
              (norm_q): LayerNorm((1792,), eps=1e-05, elementwise_affine=True)
              (norm_k): LayerNorm((1792,), eps=1e-05, elementwise_affine=True)
              (to_q): Linear(in_features=1792, out_features=1792, bias=False)
              (to_k): Linear(in_features=1792, out_features=1792, bias=False)
              (to_v): Linear(in_features=1792, out_features=1792, bias=False)
              (to_out): ModuleList(
                (0): Linear(in_features=1792, out_features=1792, bias=False)
                (1): Dropout(p=0.0, inplace=False)
              )
            )
            (feed_forward): LuminaFeedForward(
              (linear_1): Linear(in_features=1792, out_features=4864, bias=False)
              (linear_2): Linear(in_features=4864, out_features=1792, bias=False)
              (linear_3): Linear(in_features=1792, out_features=4864, bias=False)
              (silu): FP32SiLU()
            )
            (norm1): LuminaRMSNormZero(
              (silu): SiLU()
              (linear): Linear(in_features=1024, out_features=7168, bias=True)
              (norm): RMSNorm()
            )
            (ffn_norm1): RMSNorm()
            (norm2): RMSNorm()
            (ffn_norm2): RMSNorm()
            (norm1_context): RMSNorm()
          )
        )
        (norm_out): LuminaLayerNormContinuous(
          (silu): SiLU()
          (linear_1): Linear(in_features=1024, out_features=1792, bias=True)
          (norm): LayerNorm((1792,), eps=1e-06, elementwise_affine=False)
          (linear_2): Linear(in_features=1792, out_features=1792, bias=True)
        )
      )
    )
  )
  (lm_head): Linear(in_features=3584, out_features=151668, bias=False)
)
