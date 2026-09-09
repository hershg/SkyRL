"""Private vLLM 0.28 buffer inspection, only for the owned Qwen TP1 diagnostic."""

import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from examples.model_checks.active_lora_audit import (
    check_untargeted_buffers,
    compare_active_tensors,
)
from skyrl.backends.skyrl_train.inference_servers.new_inference_worker_wrap import (
    NewInferenceWorkerWrap,
)


class ActiveLoRAAuditWorker(NewInferenceWorkerWrap):
    def audit_active_lora(self, adapter_path):
        version = importlib.metadata.version("vllm")
        assert version.split("+")[0] == "0.28.0", version
        manager = self.model_runner.lora_manager._adapter_manager
        assert not manager.lora_config.fully_sharded_loras
        ids = list(manager._registered_adapters)
        assert len(ids) == 1, ids
        slots = [index for index, value in enumerate(manager.lora_index_to_id) if value == ids[0]]
        assert len(slots) == 1, manager.lora_index_to_id
        slot = slots[0]
        torch.cuda.synchronize()
        loaded = {}
        untargeted = []
        for name, module in manager.modules.items():
            if name in ("model.embed_tokens", "lm_head"):
                assert type(module).__name__ in ("VocabParallelEmbeddingWithLoRA", "LogitsProcessorWithLoRA")
                check_untargeted_buffers(name, (module.lora_a_stacked[slot], module.lora_b_stacked[slot]))
                untargeted.append(name)
                continue
            assert type(module).__name__ in (
                "RowParallelLinearWithLoRA",
                "QKVParallelLinearWithLoRA",
                "MergedColumnParallelLinearWithLoRA",
            ), (name, type(module).__name__)
            assert module.tp_size == 1
            for label, buffers in (("A", module.lora_a_stacked), ("B", module.lora_b_stacked)):
                for index, buffer in enumerate(buffers):
                    assert buffer.dtype == torch.bfloat16, (name, buffer.dtype)
                    loaded[name, index, label] = buffer[slot, 0].detach().cpu()
        path = Path(adapter_path)
        config = json.loads((path / "adapter_config.json").read_text())
        assert config["r"] == config["lora_alpha"] == 32
        assert not config.get("use_rslora", False) and not config.get("use_dora", False)
        weights = path / "adapter_model.safetensors"
        exported = load_file(weights)
        assert len(exported) == 392 and all(t.dtype == torch.float32 for t in exported.values())
        result = compare_active_tensors(exported, loaded, config["r"], config["lora_alpha"])
        with weights.open("rb") as stream:
            result["export_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        result.update(vllm_version=version, adapter_id=ids[0], slot=slot, dtype="bfloat16", tp=1, zero_untargeted=untargeted)
        return result
