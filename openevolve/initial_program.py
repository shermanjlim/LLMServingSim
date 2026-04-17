# Evolve target: MemoryModel.apply_kv_cache_events from
# inference_serving/memory_model.py.
#
# The evaluator extracts the function below (between the EVOLVE-BLOCK markers),
# re-indents it to method-level, and patches it into memory_model.py before
# running the LLMServingSim command. Keep the signature
# `def apply_kv_cache_events(self):` so the patch can locate the method.
#
# Available on `self` (see MemoryModel.__init__):
#   self.npu_prefix_cache, self.second_tier_prefix_cache  # RadixCache
#   self.npu_mem, self.npu_used, self.hbf_mem, self.hbf_used  # bytes
#   self.npu_floor, self.hbf_floor                           # post-weight floors
#   self.npu_num, self.prefix_storage, self.enable_hbf_kv,
#   self.enable_prefix_sharing, self.logger
#   self._npu_cache_hashtolen, self._cpu_cache_hashtolen,
#   self._block_hash_to_device, self._bytes_per_token
#   self.get_kv(seq), self.allocate(size, device), self.free(size, device)
# Names imported at module scope of memory_model.py:
#   BlockStored, BlockRemoved (from .radix_tree)
#   Device (NPU / CPU / CXL / HBF enum)
# BlockStored fields useful for placement:
#   ev.block_hash, ev.parent_block_hash, ev.token_ids, ev.block_size, ev.lora_id
#   ev.full_token_ids  -- full token prefix from root up to end of this block
#   ev.hit_count       -- historical match_prefix hits for this block_hash
#                         (non-zero means this content was matched before eviction)
# RadixCache also exposes `npu_prefix_cache.block_hit_counts[block_hash]` for
# querying live hit counts of any tracked block, updated on every match_prefix.
#
# Optimization goal: increase (or hold) steady_state_total_token_throughput_tok_per_s
# while decreasing steady_state_hbf_write_rate_MBps. HBF writes are tracked inside
# self.allocate(..., Device.HBF), so reducing how often / how much you push into
# HBF — without starving the NPU prefix cache — is the lever.

# EVOLVE-BLOCK-START
def apply_kv_cache_events(self):
    npu_byte_alloc = 0
    npu_byte_free = 0
    hbf_byte_alloc = 0
    hbf_byte_free = 0
    cpu_byte_alloc = 0
    cpu_byte_free = 0
    for ev in self.npu_prefix_cache.take_events():
        if isinstance(ev, BlockStored):
            tlen = len(ev.token_ids)
            h = ev.block_hash
            if h in self._npu_cache_hashtolen:
                raise RuntimeError("hash collision!")
            self._npu_cache_hashtolen[h] = tlen
            kv_bytes = self.get_kv(tlen)
            # TODO: improve HBF allocation policy
            if self.enable_hbf_kv:
                npu_free = self.npu_mem - self.npu_used - (npu_byte_alloc - npu_byte_free)
                if kv_bytes <= npu_free:
                    npu_byte_alloc += kv_bytes
                    self._block_hash_to_device[h] = Device.NPU
                else:
                    hbf_byte_alloc += kv_bytes
                    self._block_hash_to_device[h] = Device.HBF
            else:
                npu_byte_alloc += kv_bytes
        elif isinstance(ev, BlockRemoved):
            h = ev.block_hash
            tlen = self._npu_cache_hashtolen.pop(h, 0)
            if tlen == 0:
                self.logger.warning("NPU prefix cache remove unknown block hash {h}")
                continue
            kv_bytes = self.get_kv(tlen)
            if self.enable_hbf_kv:
                device = self._block_hash_to_device.pop(h)
                if device == Device.HBF:
                    hbf_byte_free += kv_bytes
                else:
                    npu_byte_free += kv_bytes
            else:
                npu_byte_free += kv_bytes

    if npu_byte_free > 0:
        self.free(npu_byte_free, Device.NPU)
    if npu_byte_alloc > 0:
        self.allocate(npu_byte_alloc, Device.NPU)
    if hbf_byte_free > 0:
        self.free(hbf_byte_free, Device.HBF)
    if hbf_byte_alloc > 0:
        self.allocate(hbf_byte_alloc, Device.HBF)

    if not self.enable_prefix_sharing and self.prefix_storage is Device.CPU:
        for ev in self.second_tier_prefix_cache.take_events():
            if isinstance(ev, BlockStored):
                tlen = len(ev.token_ids)
                self._cpu_cache_hashtolen[ev.block_hash] = tlen
                cpu_byte_alloc += self.get_kv(tlen) * self.npu_num
            elif isinstance(ev, BlockRemoved):
                tlen = self._cpu_cache_hashtolen.pop(ev.block_hash, 0)
                cpu_byte_free += self.get_kv(tlen) * self.npu_num

        if cpu_byte_alloc > 0:
            self.allocate(cpu_byte_alloc, Device.CPU)
        if cpu_byte_free > 0:
            self.free(cpu_byte_free, Device.CPU)
# EVOLVE-BLOCK-END
