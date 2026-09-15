# RTX 5090 - Qwen3.8 MIXED serve recipe

**Validated seat** for [`AEON-7/Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-NVFP4-MIXED`](https://huggingface.co/AEON-7/Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-NVFP4-MIXED) on **NVIDIA RTX 5090** (sm_120, 32 GB dedicated).

This is **not** the DGX Spark recipe. Different SM, different image, different speculative path.

| | Spark (repo root) | **This folder (5090)** |
|---|---|---|
| Image | `aeon-vllm-ultimate:2026-09-11-v0.29.0-omni` | **`aeon-vllm-ultimate-rtx:latest`** |
| Spec | Dynamic DFlash lattice | **MTP n=3** (in-checkpoint) |
| Drafter mount | Required (`z-lab/Qwen3.8-27B-DFlash2`) | **None** - do not hang DFlash on 32 GB |
| Context | 262144 | **131072** |
| Seqs / util | 16 / 0.80 | **4 / 0.92** (0.95 if KV tight) |
| KV | `fp8` | **`fp8_e4m3`** |
| Attention | `TRITON_ATTN` | **`TRITON_ATTN`** (FlashInfer+fp8 KV can garbage on sm_120) |
| Quant flag | **unset** | **unset** |

## QuickStart

```bash
cd other-hardware/rtx5090
# Weights: prefer repo-root models/ or HF cache
hf download AEON-7/Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-NVFP4-MIXED \
  --local-dir ../../models/aeon-mixed

docker compose up -d
docker compose logs -f vllm
```

Smoke:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"aeon","messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

## ModelOpt #54367

- **Optional** on `:latest` / `:2026-09-07-omni-mm` (folded in).
- **Required** on older Aug-21 RTX images - symptom: `'MergedColumnParallelLinear' object has no attribute 'data'`.
- Bind path on RTX is **`dist-packages`**, not Spark's `site-packages`.

## Do not

- Do not use DFlash / mount a fat drafter on 32 GB for the long seat.
- Do not set `--quantization`.
- Do not use YaRN / 1M on this recipe.
- Do not use the Spark (`aeon-vllm-ultimate`) image on 5090.
- Do not set `repetition_penalty` > 1.0.

Canonical docker block: [HF MIXED § RTX 5090](https://huggingface.co/AEON-7/Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-NVFP4-MIXED).
