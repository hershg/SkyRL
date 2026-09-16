"""Read physical receiver state through the pinned vLLM public buffer layout."""

import os

import torch
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.lora.layers.base_linear import BaseLinearLayerWithLoRA
from vllm.lora.layers.fused_moe import FusedMoEWithLoRA

from examples.model_checks.tensor_checks import describe_tensor, describe_tensor_storage
from skyrl.backends.skyrl_train.inference_servers.new_inference_worker_wrap import (
    NewInferenceWorkerWrap,
)


def get_active_buffers(manager):
    if len(manager.lora_index_to_id) != 1 or manager.lora_index_to_id[0] is None:
        raise ValueError("Expected exactly one active receiver slot")
    adapter_id = manager.lora_index_to_id[0]
    if adapter_id <= 0:
        raise ValueError("Receiver adapter identity must be positive")
    tensors = {}
    for name, module in sorted(manager.modules.items()):
        if isinstance(module, FusedMoEWithLoRA):
            if module.adapter_enabled[0].item() != 1:
                raise ValueError("MoE receiver adapter is disabled")
            fields = ("w13_lora_a_stacked", "w13_lora_b_stacked", "w2_lora_a_stacked", "w2_lora_b_stacked")
        elif isinstance(module, BaseLinearLayerWithLoRA):
            fields = ("lora_a_stacked", "lora_b_stacked")
        else:
            raise ValueError(f"Unsupported receiver wrapper: {type(module).__name__}")
        for field in fields:
            buffers = getattr(module, field)
            if not buffers:
                raise ValueError(f"Missing receiver factors: {name}.{field}")
            for index, buffer in enumerate(buffers):
                tensor = buffer[0]
                if not tensor.numel() or tensor.dtype != torch.bfloat16 or tensor.device.type != "cuda":
                    raise ValueError("Receiver factors must be nonempty BF16 CUDA tensors")
                tensors[f"{name}.{field}.{index}"] = tensor
    if not tensors:
        raise ValueError("Active receiver adapter has no buffers")
    return adapter_id, tensors


class H1ReceiverAuditWorker(NewInferenceWorkerWrap):
    def evict_active_lora(self):
        manager = self.model_runner.get_model().lora_manager
        if len(manager.lora_index_to_id) != 1 or manager.lora_index_to_id[0] is None:
            raise ValueError("Expected one active adapter before eviction")
        adapter_id = manager.lora_index_to_id[0]
        if not self.model_runner.remove_lora(adapter_id):
            raise ValueError("Receiver adapter removal failed")
        if self.model_runner.list_loras() or any(item is not None for item in manager.lora_index_to_id):
            raise ValueError("Receiver adapter remains registered or active")
        return {"adapter_id": adapter_id, "tp_rank": get_tensor_model_parallel_rank(), "evicted": True}

    def describe_active_lora(self):
        adapter_id, tensors = get_active_buffers(self.model_runner.get_model().lora_manager)
        buffers = {name: describe_tensor(tensor) for name, tensor in tensors.items()}
        if not buffers or {value["dtype"] for value in buffers.values()} != {"torch.bfloat16"}:
            raise ValueError("Expected BF16 receiver buffers on every rank")
        parameters = list(self.model_runner.model.named_parameters())
        model_dtype = self.model_config.dtype
        if model_dtype != torch.bfloat16 or not parameters:
            raise ValueError("Expected an effective BF16 base model")
        if any(
            parameter.is_floating_point()
            and parameter.dtype != torch.bfloat16
            and not (name.endswith(".e_score_correction_bias") and parameter.dtype == torch.float32)
            for name, parameter in parameters
        ):
            raise ValueError("Base-model parameters include a non-BF16 floating tensor")
        kv_tensors = list(self.model_runner.kv_caches)
        if not kv_tensors or any(tensor.dtype != torch.bfloat16 for tensor in kv_tensors):
            raise ValueError("Actual KV tensors must be nonempty BF16")
        context = self.model_config.max_model_len
        if context != 32768:
            raise ValueError("Effective inference context must be 32768")
        return {
            "adapter_id": adapter_id,
            "process_id": os.getpid(),
            "tp_rank": get_tensor_model_parallel_rank(),
            "buffers": buffers,
            "model_dtype": str(model_dtype),
            "model_parameter_count": len(parameters),
            "context": context,
            "kv_tensors": [describe_tensor_storage(tensor) for tensor in kv_tensors],
            "parameter_dtypes": sorted({str(parameter.dtype) for _, parameter in parameters}),
        }
