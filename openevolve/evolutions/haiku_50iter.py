from .memory_model import Device

# EVOLVE-BLOCK-START
def _device_allocate_policy(self, ev, kv_bytes, npu_free_bytes, hbf_free_bytes):
    """Decide whether a newly allocated KV-cache block lands on NPU HBM or HBF.

    Args:
        self: The enclosing ``MemoryModel`` instance. Useful read-only
            attributes the policy may consult:

            * ``self.npu_mem`` (int): total NPU HBM capacity, in bytes.
            * ``self.hbf_mem`` (int): total HBF capacity, in bytes.
            * ``self.npu_used`` (int): NPU bytes currently allocated
              (weights + KV).
            * ``self.hbf_used`` (int): HBF bytes currently allocated
              (weights + KV).
            * ``self.npu_floor`` (int): NPU bytes reserved for weights;
              KV eviction cannot free below this.
            * ``self.hbf_floor`` (int): HBF bytes reserved for weights.
            * ``self.block_size`` (int): tokens per cache block
              (e.g. 16).
            * ``self.hbf_write_bytes`` (int), ``self.hbf_write_count``
              (int): cumulative HBF-write statistics so far.
            * ``self.get_kv(tokens)`` -> int: bytes occupied by KV
              for ``tokens`` tokens on a single rank.

        ev (BlockStored): the cache event triggering this placement
            decision. Relevant fields:

            * ``ev.block_hash`` (int): hash of this block's tokens.
            * ``ev.token_ids`` (list[int]): the tokens stored in this
              block. ``len(ev.token_ids) == ev.block_size`` normally
              (the last block of a sequence may be shorter).
            * ``ev.block_size`` (int): cache-manager block size
              (matches ``self.block_size``).
            * ``ev.full_token_ids`` (list[int]): every token from the
              radix root down to and including this block's tokens.
              A longer list means the block sits deeper in a shared
              prefix.
            * ``ev.hit_count`` (int): historical reuse count tracked
              by the radix cache for this block hash. Non-zero means
              this exact block content has been matched before
              (possibly evicted and now being re-stored).
            * ``ev.num_input_tokens`` (int): original prompt length
              (``req.original_input``) of the request that caused this
              block to be stored. Useful for distinguishing prefill
              blocks from blocks produced by long decodes.
            * ``ev.num_output_tokens`` (int): target total sequence
              length — input + output tokens — (``req.output``) of the
              request that caused this block to be stored. Subtract
              ``ev.num_input_tokens`` to get the pure decode length.
              Useful for anticipating how many more decode blocks this
              request will still generate.

        kv_bytes (int): bytes this single block will occupy once
            allocated (equal to ``self.get_kv(len(ev.token_ids))``).
        npu_free_bytes (int): NPU bytes free AFTER any placements
            already decided earlier in the same
            ``apply_kv_cache_events`` pass. Treat this as the ground
            truth for "can this block fit on NPU right now".
        hbf_free_bytes (int): HBF bytes free, same accounting as
            ``npu_free_bytes``.

    Returns:
        Device: exactly ``Device.NPU`` or ``Device.HBF``. Any other
        value raises a ``RuntimeError`` in the caller.
    """
    # Hot blocks go to NPU
    if ev.hit_count > 0:
        return Device.NPU if kv_bytes <= npu_free_bytes else Device.HBF

    # Cold blocks: HBF-first strategy to reduce write amplification
    # However, if HBF is filling up (>85%), prefer NPU to preserve space for hot blocks
    hbf_capacity = self.hbf_mem - self.hbf_floor
    if hbf_capacity > 0:
        hbf_util = (hbf_capacity - hbf_free_bytes) / hbf_capacity
        # More aggressive threshold + deeper prefixes get NPU priority
        if hbf_util > 0.85 and kv_bytes <= npu_free_bytes:
            return Device.NPU
        # Deep prefixes (>80 tokens, ~5 blocks) in moderate HBF pressure favor NPU
        if hbf_util > 0.75 and len(ev.full_token_ids) > 80 and kv_bytes <= npu_free_bytes:
            return Device.NPU

    if kv_bytes <= hbf_free_bytes:
        return Device.HBF

    # Fallback to NPU if HBF full
    return Device.NPU if kv_bytes <= npu_free_bytes else Device.HBF
# EVOLVE-BLOCK-END
