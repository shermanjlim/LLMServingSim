def _device_allocate_policy(self, ev, kv_bytes, npu_free_bytes, hbf_free_bytes):
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