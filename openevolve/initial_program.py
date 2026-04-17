"""
Evolve target for LLMServingSim KV-cache placement policy.

Only the function between the EVOLVE-BLOCK markers is patched into
`inference_serving/memory_model.py` — the rest of this file is context for
the LLM. The evaluator (openevolve/evaluator.py) runs a full simulation and
scores the policy on `throughput / (1 + hbf_write_rate_MBps)`, with a hard
constraint that steady-state throughput must not fall below the current
baseline. Do not change the signature or return type of the function.

================================================================
WHAT THE POLICY DOES
================================================================
`_device_allocate_policy` is a *placement* decision for prefix-cache blocks
in the radix cache (RadixCache). It is called exactly once per `BlockStored`
event emitted by `self.npu_prefix_cache` inside `apply_kv_cache_events`
(see memory_model.py). The prefix cache is the primary driver of hit-rate
improvements: every hit avoids a recompute during prefill, which raises
throughput.

When `self.enable_hbf_kv` is True, the cache spans NPU + HBF, and this
policy chooses which device backs the newly stored block. The function MUST
return exactly `Device.NPU` or `Device.HBF`. Anything else raises.

Returning a device with insufficient free bytes will raise inside
`MemoryModel.allocate` — so the policy MUST fall back to the other device
when its first choice would overflow. The radix cache has already evicted
enough total capacity before emitting the event, so at least one of
`npu_free_bytes` and `hbf_free_bytes` is >= `kv_bytes`.

================================================================
FUNCTION ARGUMENTS
================================================================
`self`   — the enclosing MemoryModel. Useful attributes you may read:
            * self.npu_mem, self.hbf_mem            (total capacity, bytes)
            * self.npu_used, self.hbf_used          (current bytes in use)
            * self.npu_floor, self.hbf_floor        (reserved for weights)
            * self.block_size                        (tokens per block, e.g. 16)
            * self.n_layer, self.fp, self.kv_dim     (model shape)
            * self._block_hash_to_device             (dict hash -> Device for
                                                      previously placed blocks;
                                                      useful for parent lookup)
            * self._npu_cache_hashtolen              (hash -> block length for
                                                      all currently cached blocks)
            * self.get_kv(tokens)                    (bytes for N tokens of KV)
            * self.hbf_write_bytes / count           (running write stats)

`ev`     — a BlockStored event (see inference_serving/radix_tree.py). Fields:
            * ev.block_hash          : int   — hash of this block's tokens.
            * ev.token_ids           : list[int] — tokens stored in this block
                                                    (len == block_size normally).
            * ev.block_size          : int   — cache-manager block size.
            * ev.full_token_ids      : list[int] — every token from the radix
                                                    root down to and including
                                                    this block (prefix length).
                                                    A long full_token_ids
                                                    implies the block sits
                                                    deep in a shared prefix.
            * ev.hit_count           : int   — historical reuse count for this
                                                block hash *tracked by the
                                                cache*. Non-zero means this
                                                exact block content has been
                                                matched before (possibly
                                                evicted and now being
                                                re-stored). A strong signal of
                                                expected reuse / hotness.

`kv_bytes`       — bytes this single block will occupy once allocated.
`npu_free_bytes` — NPU bytes free AFTER any pending allocations queued earlier
                    in the same `apply_kv_cache_events` pass. Safe to use as
                    the ground truth "can this block fit on NPU right now".
`hbf_free_bytes` — same, for HBF.

================================================================
SIMULATED SYSTEM (our_cluster_config/6_hbm_2_hbf.json, Llama-3.1-8B, fp16)
================================================================
* NPU HBM : 24 GB, 768 GB/s, 0 ns latency.
* HBF     : 80 GB, 768 GB/s, 10_000 ns latency (~10 µs per access).
* Weights (~16 GB in fp16) live on HBF in this run (enable_hbf_offload=True),
  so `npu_floor = 0` and effectively the entire 24 GB NPU HBM is free for KV.
* block_size = 16 tokens. One block of KV ≈ 16 * 2 * kv_dim * n_layer * fp/8
  bytes per NPU. For Llama-3.1-8B (n_layer=32, kv_head=8, head_dim=128,
  fp16) → one block ≈ 256 KiB of KV.
* Prefix cache capacity = (npu_mem - npu_used_weights) + (hbf_mem - hbf_used_weights)
  ≈ 24 GB + 64 GB. The radix cache evicts on its own when full.

================================================================
HARD CONSTRAINTS (violating any of these will either raise at runtime or
disqualify the candidate via an error in the evaluator)
================================================================
  1. Return exactly `Device.NPU` or `Device.HBF`. No other value.
  2. If you return NPU, `kv_bytes` MUST be <= `npu_free_bytes`.
     If you return HBF, `kv_bytes` MUST be <= `hbf_free_bytes`.
     When the first choice doesn't fit, fall back to the other device.
  3. Keep the function pure and fast — it runs inside a tight inner loop.
     Do not perform I/O, do not mutate `self`, do not modify `ev`.
  4. Do not import modules that aren't already imported at the top of
     `inference_serving/memory_model.py` (Device, logging, os, threading,
     and everything from radix_tree). If you need math, use arithmetic.

================================================================
BASELINE TO BEAT (our_cluster_config/6_hbm_2_hbf.json, 2000 requests)
================================================================
The trivial "NPU first, HBF on overflow" policy below achieves, in steady
state:
    throughput  ≈ 5646 tok/s
    hbf_write   ≈ 407 MB/s
    score       ≈ 13.88

Target: keep throughput >= 5646 tok/s, drive `hbf_write_rate_MBps` downward.
"""

# EVOLVE-BLOCK-START
def _device_allocate_policy(self, ev, kv_bytes, npu_free_bytes, hbf_free_bytes):
    # Baseline placement: fill NPU first, spill to HBF when it can't fit.
    # Constraints reiterated for the evolved variants:
    #   - Must return Device.NPU or Device.HBF.
    #   - Must not return a device where kv_bytes > <that device>_free_bytes.
    # Ideas to explore (read from `self` and `ev`):
    #   - Use ev.hit_count as a hotness signal (>0 implies past reuse).
    #   - Inspect self._block_hash_to_device.get(ev.parent_block_hash) to see
    #     where the parent block landed; keep chains together.
    #   - Reserve a fraction of NPU (e.g. 20%) as headroom for per-request KV,
    #     sending cold blocks to HBF earlier than the hard capacity limit.
    #   - Use len(ev.full_token_ids) as a proxy for prefix depth / value.
    if kv_bytes <= npu_free_bytes:
        return Device.NPU
    return Device.HBF
# EVOLVE-BLOCK-END
