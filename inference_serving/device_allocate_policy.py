from .memory_model import Device

# EVOLVE-BLOCK-START
def _device_allocate_policy(self, events, npu_free_bytes, hbf_free_bytes):
    """Decide whether each newly allocated KV-cache block lands on NPU HBM or HBF.

    Called once per ``apply_kv_cache_events`` pass with all the
    ``BlockStored`` events produced since the previous pass. All
    ``BlockRemoved`` events in the same pass have already been applied
    to ``self.npu_used``/``self.hbf_used`` before this call, so
    ``npu_free_bytes`` and ``hbf_free_bytes`` already reflect those
    frees.

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

        events (list[BlockStored]): the cache events triggering these
            placement decisions. One decision must be produced per
            event, in the same order. Relevant fields on each event:

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

        npu_free_bytes (int): NPU bytes free at the start of this
            policy call (after all BlockRemoved events in this pass
            have been applied). The policy is responsible for
            deducting bytes as it places each event if it wants
            per-event free accounting.
        hbf_free_bytes (int): HBF bytes free, same accounting as
            ``npu_free_bytes``.

    Returns:
        list[Device]: one ``Device.NPU`` or ``Device.HBF`` entry per
        event, in the same order as ``events``. Any other value raises
        a ``RuntimeError`` in the caller.
    """
    decisions = []
    for ev in events:
        kv_bytes = self.get_kv(len(ev.token_ids))
        if kv_bytes <= npu_free_bytes:
            decisions.append(Device.NPU)
            npu_free_bytes -= kv_bytes
        else:
            decisions.append(Device.HBF)
            hbf_free_bytes -= kv_bytes
    return decisions
# EVOLVE-BLOCK-END
