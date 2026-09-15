# AGENTS.md - Operator's Manual for AI Agents (Qwen3.8 MIXED)

**Read this file before changing anything in this repo or its container.**

You are an AI coding agent working with this repository, its container images, or the model it serves. Public docs, blog posts, and even Qwen3.6 recipes in sibling repos are **stale for this stack**. Qwen3.8 **NVFP4-MIXED** has different quantization, attention, speculative, and util locks than Qwen3.6 compressed-tensors / flash_attn recipes.

This file is authoritative for **Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-NVFP4-MIXED**. If public docs contradict it, **trust this file**. Canonical docker blocks also live on the [HF MIXED card](https://huggingface.co/AEON-7/Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-NVFP4-MIXED).

---

## ⚠️ Hardware scope

**Root compose + most of this file target DGX Spark / GB10 / sm_121a (UMA).** RTX seats live under `other-hardware/` with different images and speculative paths.

| You're on | Recipe location | Why Spark rules don't apply wholesale |
|---|---|---|
| **1x DGX Spark** (GB10, sm_121a) | [`docker-compose.yml`](docker-compose.yml) | This file's defaults |
| **2x Spark TP=2** | [`docker-compose.tp2-rank1.yml`](docker-compose.tp2-rank1.yml) -> [`docker-compose.tp2-rank0.yml`](docker-compose.tp2-rank0.yml) | Fixed DFlash2 n=7 + YaRN; MRv2 **off**; B0 allreduce knobs |
| **RTX 5090** (sm_120, 32 GB) | [`other-hardware/rtx5090/`](other-hardware/rtx5090/) | RTX image; **MTP n=3**; no DFlash; util 0.92 |
| **RTX PRO 6000** (sm_120, 96 GB) | [`other-hardware/rtx6000pro/`](other-hardware/rtx6000pro/) | Same RTX image; MTP n=3; util 0.80; **validated for Qwen3.8** |
| **BF16 / H200 teacher** | HF BF16 card only | Not day-to-day serve - no compose in this repo |

Do **not** apply Spark util/DFlash/image advice to RTX, or RTX MTP advice to Spark.

---

## TL;DR for agents (60 seconds)

| Thing | Value | Don't second-guess |
|---|---|---|
| **Body** | `AEON-7/Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-NVFP4-MIXED` (~23.8G) | Not the Qwen3.6 tree; not compressed-tensors-only dumps |
| **Spark image** | `ghcr.io/aeon-7/aeon-vllm-ultimate:2026-09-11-v0.29.0-omni` (digest `sha256:2421bb...`) | Pin dated until `:latest` matches; rollback `:2026-09-07-reasoning-eos`. ENTRYPOINT is bash -> compose sets `entrypoint: vllm` |
| **RTX image** | `ghcr.io/aeon-7/aeon-vllm-ultimate-rtx:latest` | Do **not** cross Spark ↔ RTX images |
| **Quantization** | **Leave `--quantization` UNSET** | `hf_quant_config.json` -> `modelopt_mixed`. Never `compressed-tensors` / `nvfp4` / `modelopt` / `modelopt_fp4` on this tree |
| **Attention** | **`--attention-backend TRITON_ATTN`** | Never `flash_attn` on these MIXED recipes |
| **Prefix cache** | **`--no-enable-prefix-caching`** | Always off on published MIXED seats |
| **Spark spec** | **Dynamic DFlash lattice** + `z-lab/Qwen3.8-27B-DFlash2` | Exact map below; peak **237.67** tok/s Coding@c16 |
| **TP=2 spec** | Fixed **DFlash2 n=7** | Not the lattice |
| **RTX spec** | **MTP n=3** | No DFlash on 5090 |
| **Spark util** | **0.80** default | Downshift if sidecars; optional **0.85** dedicated-only; beyond 0.80 -> UMA OOM risk |
| **#54367** | Bind `patches/modelopt-54367.py` on Spark 0.29 | **Required** for MIXED on `2026-09-11-v0.29.0-omni` |
| **Gen defaults** | temp 0.6, top_p 0.95, top_k 20, **`repetition_penalty` 1.0** | **>1.0 breaks `/parameter` / XML tool parsers** |
| **Served name** | `aeon` (alias `aeon-ultimate` OK) | |
| **Chat** | `reasoning-parser qwen3`, `tool-call-parser qwen3_coder`, `enable-auto-tool-choice` | |

Compose files are the **source of truth**. Don't invent flags.

### Dynamic DFlash lattice (1x Spark - EXACT)

```json
{"method":"dflash","model":"/draft","num_speculative_tokens":10,"num_speculative_tokens_per_batch_size":[[1,1,10],[2,2,10],[3,4,8],[5,8,7],[9,10,6],[11,12,5],[13,14,4],[15,16,3]],"attention_backend":"TRITON_ATTN"}
```

Runtime K: c1-2->10, c3-4->8, c5-8->7, c9-10->6, c11-12->5, c13-14->4, c15-16->3.

---

## Pick-the-compose guide

| Goal | File(s) | Notes |
|---|---|---|
| Production 1x Spark (quality + throughput) | `docker-compose.yml` | Lattice, util 0.80, 262k, MRv2 on, FULL_AND_PIECEWISE |
| Dual Spark 1M quality | `docker-compose.tp2-rank1.yml` **then** `docker-compose.tp2-rank0.yml` | Rank 1 headless first; YaRN factor-4; util 0.70 |
| Dual Spark 64k speed | Same TP2 files with deltas in comments | Drop YaRN / long-len env; max-model-len 65536; util 0.60 |
| RTX 5090 chat | `other-hardware/rtx5090/docker-compose.yml` | MTP n=3; util 0.92 |
| RTX PRO 6000 | `other-hardware/rtx6000pro/docker-compose.yml` | MTP n=3; util 0.80; validated Qwen3.8 |

---

## DO NOT UNDO - critical traps (Qwen3.8 MIXED)

### 1. Don't set `--quantization`

```
# WRONG for MIXED:
--quantization compressed-tensors
--quantization nvfp4
--quantization modelopt
--quantization modelopt_fp4
```

**Why:** MIXED needs `hf_quant_config.json` to select **`modelopt_mixed`**. Passing an explicit quant flag forces the wrong loader. Leave it **unset**.

### 2. Don't use `flash_attn` - must `TRITON_ATTN`

```
# WRONG (stale Qwen3.6 Spark advice):
--attention-backend flash_attn
```

**Why:** Published Qwen3.8 MIXED seats (Spark + RTX long) require **`TRITON_ATTN`**. On RTX sm_120, FlashInfer attention + fp8 KV can emit **garbage tokens**.

### 3. Don't enable prefix caching on these recipes

```
# WRONG:
--enable-prefix-caching
```

**Why:** All published MIXED seats use **`--no-enable-prefix-caching`**. Qwen3.6 Spark production may enable prefix cache under different patches - that advice does **not** transfer.

### 4. Don't use DFlash on 5090 (MTP only)

Fat DFlash2 (~drafter GB) steals the 32 GB KV pool. Use in-checkpoint **MTP n=3**.

### 5. Don't use MTP on Spark (DFlash lattice / TP2 n=7)

Spark published path is external DFlash. Never MTP + DFlash together.

### 6. Don't mix Spark and RTX images

| Arch | Image |
|---|---|
| sm_121a / aarch64 UMA | `aeon-vllm-ultimate:2026-09-11-v0.29.0-omni` |
| sm_120 / amd64 dedicated | `aeon-vllm-ultimate-rtx:latest` |

### 7. Don't raise Spark util above 0.80 without understanding UMA OOM

Default **0.80**. Downshift with secondary services. Optional **0.85** dedicated-only. Beyond 0.80 risks OOM under high KV.

### 8. Don't set `repetition_penalty` > 1.0

Values above 1.0 penalize repeated structural tokens (`/parameter`, XML tags) and **break tool / structured parsers**. Lock **1.0**.

### 9. Don't forget #54367 bind on Spark 0.29 MIXED

In-tree 0.29 ModelOpt alone does **not** load this lattice. Mount `patches/modelopt-54367.py` over:

`/usr/local/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/modelopt.py`

(RTX uses `dist-packages`; optional on RTX `:latest`.)

### 10. Don't use Qwen3.6 drafter / compressed-tensors body

Drafter: **`z-lab/Qwen3.8-27B-DFlash2`** (not `Qwen3.6-27B-DFlash`). Body: MIXED repo above - not Qwen3.6 NVFP4 / XS trees.

### Bonus traps

- Don't invent lattice maps - use the exact JSON above.
- Don't use single-Spark lattice on TP=2 (TP=2 = fixed n=7).
- Don't start TP=2 rank 0 before rank 1.
- Don't `pip install` into the container (overwrites patched vLLM).
- Don't mount only `.../snapshots/<rev>` unless every symlink resolves inside the container - prefer HF cache root + repo-id.

---

## Required environment variables (1x Spark)

| Variable | Value | Why |
|---|---|---|
| `VLLM_USE_V2_MODEL_RUNNER` | `1` | MRv2 for lattice seat + FULL_AND_PIECEWISE |
| `VLLM_ENABLE_CUDA_COMPATIBILITY` | `0` | Published MIXED Spark seat |
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | `1` | 262k (and TP2 1M) |
| `TORCH_CUDA_ARCH_LIST` | `12.1a` | GB10 |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Fragmentation under long KV |
| `NVIDIA_FORWARD_COMPAT` / `NVIDIA_DISABLE_REQUIRE` | `1` | GB10 driver shim |
| `ENABLE_NVFP4_SM100` | `0` | sm_121a import guard |
| `VLLM_USE_FLASHINFER_MOE_FP4` | `0` | Dense model - skip MoE probe spam |
| `VLLM_USE_FLASHINFER_SAMPLER` | `1` | Faster sampling |

### TP=2 extras (B0)

| Variable / flag | Value |
|---|---|
| `VLLM_USE_V2_MODEL_RUNNER` | **`0`** (opposite of single Spark) |
| `VLLM_ALLREDUCE_USE_FLASHINFER` | **`0`** |
| `--disable-custom-all-reduce` | set |
| `NCCL_IB_HCA` | exact-match `=rocep...:1` form (`IB_HCA==rocep1s0f0:1` in shell) |

---

## Required vLLM serve flags (1x Spark)

| Flag | Value | Why |
|---|---|---|
| *(no `--quantization`)* | unset | modelopt_mixed via hf_quant_config |
| `--attention-backend` | `TRITON_ATTN` | MIXED lock |
| `--no-enable-prefix-caching` | flag | MIXED lock |
| `--gpu-memory-utilization` | `0.80` | UMA-safe default |
| `--max-model-len` | `262144` | Published single Spark |
| `--max-num-seqs` | `16` | Lattice peak-16 map |
| `--max-num-batched-tokens` | `16384` | Published |
| `--kv-cache-dtype` | `fp8` | Single Spark |
| `--enable-chunked-prefill` | flag | Long ctx |
| `--compilation-config` | `{"cudagraph_mode":"FULL_AND_PIECEWISE"}` | With MRv2 |
| `--speculative-config` | Dynamic DFlash lattice JSON | Exact map |
| `--reasoning-parser` | `qwen3` | Thinking |
| `--tool-call-parser` | `qwen3_coder` | Tools |
| `--enable-auto-tool-choice` | flag | Auto tools |
| `--override-generation-config` | rep **1.0** | Tool/XML safe |
| `--limit-mm-per-prompt` | `{"image":4,"video":2}` | MM caps |
| `--served-model-name` | `aeon` | (+ `aeon-ultimate` alias OK) |

---

## Knob cheat-sheet (all seats)

| Seat | max-model-len | seqs | util | Spec | YaRN |
|---|---:|---:|---:|---|---|
| **1x Spark** | 262144 | 16 | **0.80** | Dynamic DFlash lattice | off |
| **TP=2** | 1000000 | 16 | 0.70 | DFlash2 n=7 | on (factor-4) |
| **RTX 5090** | 131072 | 4 | 0.92-0.95 | MTP n=3 | off |
| **RTX PRO 6000** | 262144 | 8 | 0.80 | MTP n=3 | off |

---

## Common failures

| What you see | Likely cause | Fix |
|---|---|---|
| ModelOpt / mixed load error / missing attr on `MergedColumnParallelLinear` | Missing #54367 on Spark 0.29 (or Aug-21 RTX) | Bind `modelopt-54367.py` (site-packages Spark / dist-packages RTX) |
| Quant / compressed-tensors error | Someone set `--quantization` | Remove it |
| Garbage tokens on 5090 | Wrong attention / KV | `TRITON_ATTN` + `fp8_e4m3` |
| CUDA OOM on Spark | util too high / sidecars | Downshift util; don't push past 0.80 casually |
| 5090 won't start KV at 131k | util 0.92 too low for that card | Try **0.95**, then seqs 2 |
| TP=2 hang / NCCL timeout | Rank 0 before rank 1, or wrong nic | Rank **1 first**; fix `IFACE` / `IB_HCA` |
| TP=2 two APIs / port in use | Rank 1 not headless | Rank 1: `--headless` only |
| Tools / XML broken | `repetition_penalty` > 1.0 | Set **1.0** |
| Dangling weight symlinks | Snapshot-only mount | HF cache root + repo-id, or materialized `--local-dir` |
| Wrong drafter / acceptance collapse | Qwen3.6 DFlash mounted | Use `z-lab/Qwen3.8-27B-DFlash2` |
| MTP+DFlash both set | Invented combo | Pick one path only |

---

## Diagnostics

```bash
# 1. Container
docker ps --filter "name=aeon-mixed" --format "{{.Image}} | {{.Status}}"

# 2. Health
curl -sf http://127.0.0.1:8000/health

# 3. Models list
curl -sf http://127.0.0.1:8000/v1/models

# 4. Smoke
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"aeon","messages":[{"role":"user","content":"Say hi in one sentence."}],"max_tokens":64}'
```

Thinking on (per request): `chat_template_kwargs={"enable_thinking": true, "reasoning_effort": "medium"}`.

With `--reasoning-parser qwen3`, clients must read **`reasoning` + `content`** (streaming: `delta.reasoning` / `delta.content`) - content-only clients look empty during think.

---

## Weights mount notes

HF Hub snapshot dirs are **symlink trees** into `blobs/`. Mounting only `.../snapshots/<rev>` often yields **dangling symlinks** in Docker.

1. **Preferred:** `-v $HOME/.cache/huggingface:/root/.cache/huggingface:ro` then serve by repo-id + `--revision`.
2. **OK:** materialized `huggingface-cli download ... --local-dir` tree (compose default `./models/aeon-mixed`).

---

## Performance baseline (published)

- Single Spark Dynamic DFlash lattice: wave-peak **237.67** tok/s Coding@c16; mothership overall ~**91.9**, Perf dial **100**.
- TP=2 1M YaRN is the long-context quality seat - slower than 64k.
- 5090 / PRO 6000: MTP n=3 (~1% KV cost); no fat drafter.

If Spark median is far below published lattice behavior, check #54367 bind, unset quant, TRITON_ATTN, and that the **lattice** JSON (not fixed n=7) is on the single-Spark seat.

---

## What you must not do

- Do not change weights / merge LoRAs / publish from this agent session.
- Do not invent quant, attention, or speculative flags.
- Do not treat Qwen3.6 AGENTS.md flags as transferable.
- If stuck after one log read + one table match, stop and ask the human.
