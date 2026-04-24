from .memory_model import Device

# EVOLVE-BLOCK-START
def _device_allocate_policy(self, events, npu_free_bytes, hbf_free_bytes):
    decisions = []
    hbf_capacity = self.hbf_mem - self.hbf_floor
    for ev in events:
        kv_bytes = self.get_kv(len(ev.token_ids))

        # Hot blocks go to NPU
        if ev.hit_count > 0:
            if kv_bytes <= npu_free_bytes:
                decisions.append(Device.NPU)
                npu_free_bytes -= kv_bytes
            else:
                decisions.append(Device.HBF)
                hbf_free_bytes -= kv_bytes
            continue

        # Cold blocks: HBF-first, but redirect to NPU under HBF pressure
        prefer_npu = False
        if hbf_capacity > 0:
            hbf_util = (hbf_capacity - hbf_free_bytes) / hbf_capacity
            if hbf_util > 0.85 and kv_bytes <= npu_free_bytes:
                prefer_npu = True
            elif hbf_util > 0.75 and len(ev.full_token_ids) > 80 and kv_bytes <= npu_free_bytes:
                prefer_npu = True

        if prefer_npu:
            decisions.append(Device.NPU)
            npu_free_bytes -= kv_bytes
        elif kv_bytes <= hbf_free_bytes:
            decisions.append(Device.HBF)
            hbf_free_bytes -= kv_bytes
        elif kv_bytes <= npu_free_bytes:
            decisions.append(Device.NPU)
            npu_free_bytes -= kv_bytes
        else:
            decisions.append(Device.HBF)
            hbf_free_bytes -= kv_bytes
    return decisions
# EVOLVE-BLOCK-END
