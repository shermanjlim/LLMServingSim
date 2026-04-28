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

        # Cold blocks: minimize HBF writes with strict NPU promotion criteria
        hbf_util = (hbf_capacity - hbf_free_bytes) / hbf_capacity if hbf_capacity > 0 else 0
        seq_len = len(ev.full_token_ids)
        promote_to_npu = (hbf_util > 0.90 and kv_bytes <= npu_free_bytes) or \
                         (seq_len > 150 and kv_bytes <= npu_free_bytes and hbf_util > 0.80)

        if promote_to_npu:
            decisions.append(Device.NPU)
            npu_free_bytes -= kv_bytes
        elif kv_bytes <= hbf_free_bytes:
            decisions.append(Device.HBF)
            hbf_free_bytes -= kv_bytes
        else:
            decisions.append(Device.NPU)
            npu_free_bytes -= kv_bytes
    return decisions
# EVOLVE-BLOCK-END
