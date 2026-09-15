# RTX PRO 6000 - Qwen3.8 MIXED serve recipe

**Validated for Qwen3.8 MIXED** on **NVIDIA RTX PRO 6000 Blackwell** (sm_120, 96 GB dedicated).

Use **`ghcr.io/aeon-7/aeon-vllm-ultimate-rtx:latest`** - **not** stock `vllm/vllm-openai`. The older Qwen3.6 folder in other repos used stock vLLM experimentally; that does **not** apply to this Qwen3.8 MIXED lattice.

| Knob | Value |
|---|---|
| max-model-len | **262144** |
| max-num-seqs | **8** |
| gpu-memory-utilization | **0.80** |
| Spec | **MTP n=3** (`attention_backend` inside speculative-config) |
| kv-cache-dtype | `fp8_e4m3` |
| mamba-ssm-cache-dtype | `bfloat16` |
| attention | `TRITON_ATTN` |
| `--quantization` | **unset** |

## QuickStart

```bash
cd other-hardware/rtx6000pro
hf download AEON-7/Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-NVFP4-MIXED \
  --local-dir ../../models/aeon-mixed

docker compose up -d
docker compose logs -f vllm
```

## vs Spark / vs 5090

- Same **RTX** image family as 5090 (sm_120 / amd64). Never the Spark image.
- More VRAM than 5090 -> higher context (262k) and concurrency (8 seqs).
- Spec is MTP, not DFlash (Spark-only lattice / TP2 n=7).
- Put `"attention_backend":"TRITON_ATTN"` **inside** `--speculative-config` - top-level flag does not always propagate to the drafter.

## #54367

Optional on `:latest` / `2026-09-07-omni-mm`; required on Aug-21. RTX bind path: `dist-packages`.

Canonical block: [HF MIXED § RTX PRO 6000](https://huggingface.co/AEON-7/Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-NVFP4-MIXED).
