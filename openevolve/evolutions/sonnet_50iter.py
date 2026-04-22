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
    # Safety check - ensure we can place somewhere
    if kv_bytes > npu_free_bytes and kv_bytes > hbf_free_bytes:
        return Device.HBF if hbf_free_bytes >= npu_free_bytes else Device.NPU
    
    # If only one device has space, use it
    if kv_bytes > npu_free_bytes:
        return Device.HBF
    if kv_bytes > hbf_free_bytes:
        return Device.NPU
    
    # Both devices have space - decide based on reuse likelihood
    # Calculate reuse score: hit_count + prefix depth bonus
    prefix_depth = len(ev.full_token_ids) if ev.full_token_ids else 0
    reuse_score = ev.hit_count + max(0, prefix_depth - 50) // 10
    
    # Calculate memory pressure (0.0 = empty, 1.0 = full)
    npu_kv_capacity = self.npu_mem - self.npu_floor
    npu_pressure = (self.npu_used - self.npu_floor) / max(1, npu_kv_capacity)
    
    # High reuse blocks prefer NPU, but adjust threshold based on pressure
    reuse_threshold = 1 + int(npu_pressure * 3)
    
    if reuse_score >= reuse_threshold:
        return Device.NPU
    else:
        return Device.HBF
# EVOLVE-BLOCK-END
