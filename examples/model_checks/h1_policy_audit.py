"""Megatron policy audits for optimizer-backed file publication."""

import hashlib
import json
import math
import os
from pathlib import Path

import torch
from megatron.core import parallel_state
from megatron.core.transformer.transformer_layer import TransformerLayer
from safetensors import safe_open

from examples.model_checks.file_io import hash_file
from examples.model_checks.tensor_checks import describe_tensor
from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    RemoteInferenceClient,
)
from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import (
    MegatronPolicyWorkerBase,
)

TARGET_MODULES = (
    "linear_q_down_proj",
    "linear_q_up_proj",
    "linear_kv_down_proj",
    "linear_kv_up_proj",
    "linear_proj",
    "linear_fc1",
    "linear_fc2",
)


def get_adapter_parameters(chunks):
    return [
        (f"chunk{chunk_index}.{name}", parameter)
        for chunk_index, chunk in enumerate(chunks)
        for name, parameter in chunk.named_parameters()
        if ".adapter." in f".{name}."
    ]


def describe_named_tensors(tensors):
    return {name: describe_tensor(tensor) for name, tensor in tensors}


def iter_state_tensors(value, prefix=""):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from iter_state_tensors(item, child)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            child = f"{prefix}.{index}" if prefix else str(index)
            yield from iter_state_tensors(item, child)


def classify_targets(names):
    coverage = {target: False for target in TARGET_MODULES}
    categories = {name: False for name in ("attention", "dense_mlp", "routed_expert", "shared_expert")}
    unexpected = []
    for name in names:
        target = name.split(".adapter.", 1)[0].rsplit(".", 1)[-1]
        if target not in coverage:
            unexpected.append(name)
            continue
        coverage[target] = True
        categories["attention"] |= target in TARGET_MODULES[:5]
        categories["routed_expert"] |= ".experts." in name and ".shared_experts." not in name
        categories["shared_expert"] |= ".shared_experts." in name
        categories["dense_mlp"] |= (
            target in TARGET_MODULES[-2:] and ".experts." not in name and ".shared_experts." not in name
        )
    return {"targets": coverage, "categories": categories, "unexpected": unexpected}


def validate_factor_inventory(chunks, moe_layers):
    expected = set()
    for index, chunk in enumerate(chunks):
        for layer_name, layer in chunk.named_modules():
            if not isinstance(layer, TransformerLayer):
                continue
            targets = [
                name
                for name, _ in layer.named_modules()
                if name.rsplit(".", 1)[-1] in TARGET_MODULES and ".adapter." not in name
            ]
            names = [f"{name}.adapter.linear_in.weight" for name in targets]
            coverage = classify_targets(names)
            is_moe = bool(moe_layers[layer.layer_number - 1])
            if not all(coverage["targets"].values()) or coverage["categories"] != {
                "attention": True,
                "dense_mlp": not is_moe,
                "routed_expert": is_moe,
                "shared_expert": is_moe,
            }:
                return False
            expected.update(
                f"chunk{index}.{layer_name}.{name}.adapter.{side}.weight"
                for name in targets
                for side in ("linear_in", "linear_out")
            )
    factors = get_adapter_parameters(chunks)
    return (
        bool(expected)
        and {name for name, _ in factors} == expected
        and all(parameter.requires_grad and parameter.numel() > 0 for _, parameter in factors)
    )


def normalize_state_key(key):
    if type(key) in (str, int):
        return f"{type(key).__name__}:{key}"
    if type(key) is tuple and key and all(isinstance(item, torch.dtype) for item in key):
        return "dtype_tuple:" + ",".join(str(item) for item in key)
    raise TypeError("Unsupported checkpoint mapping key")


def normalize_state(value):
    if isinstance(value, torch.Tensor):
        descriptor = describe_tensor(value)
        return {key: descriptor[key] for key in ("dtype", "shape", "sha256")}
    if isinstance(value, dict):
        return {
            normalize_state_key(key): normalize_state(item)
            for key, item in sorted(value.items(), key=lambda item: normalize_state_key(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [normalize_state(item) for item in value]
    if type(value) is float and not math.isfinite(value):
        raise ValueError("Nonfinite checkpoint state")
    if type(value) in (str, int, float, bool) or value is None:
        return value
    if isinstance(value, (torch.dtype, torch.device)):
        return {"type": type(value).__name__, "value": str(value)}
    raise TypeError(f"Unsupported checkpoint state: {type(value).__name__}")


def fingerprint_state(value):
    payload = json.dumps(normalize_state(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def collect_optimizer_state(optimizer):
    """Read metadata and actual local parameter/moment buffers through Core's public API."""
    optimizers = getattr(optimizer, "chained_optimizers", [optimizer])
    return [
        {
            "metadata": part.state_dict(),
            "parameter_state": part.get_parameter_state_dp_reshardable(),
        }
        for part in optimizers
    ]


def iter_adam_moments(value):
    if isinstance(value, dict):
        if "param" in value:
            parameter = value["param"]
            for key in ("exp_avg", "exp_avg_sq"):
                moment = value[key]
                if not isinstance(moment, torch.Tensor) or moment.shape != parameter.shape or not moment.numel():
                    raise ValueError("Missing or incomplete local Adam moment tensor")
                yield moment
        else:
            for item in value.values():
                yield from iter_adam_moments(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_adam_moments(item)


class PublicationAuditClient(RemoteInferenceClient):
    async def load_lora_adapter(self, lora_name, lora_path):
        self.publication_receipts = await super().load_lora_adapter(lora_name, lora_path)
        return self.publication_receipts


class H1PolicyAuditWorker(MegatronPolicyWorkerBase):
    _h1_publication_index = 0
    _h1_last_lora_path = None

    def describe_lora_factors(self):
        factors = get_adapter_parameters(self.actor_module)
        coverage = classify_targets([name for name, _ in factors])
        dtypes = {str(parameter.dtype) for _, parameter in factors}
        passed = bool(factors) and dtypes == {"torch.float32"}
        layers = sorted(
            module.layer_number
            for chunk in self.actor_module
            for module in chunk.modules()
            if isinstance(module, TransformerLayer)
        )
        moe_layers = self.provider.moe_layer_freq
        expected_categories = {
            "attention": True,
            "dense_mlp": any(moe_layers[layer - 1] == 0 for layer in layers),
            "routed_expert": any(moe_layers[layer - 1] == 1 for layer in layers),
            "shared_expert": any(moe_layers[layer - 1] == 1 for layer in layers),
        }
        passed &= validate_factor_inventory(self.actor_module, moe_layers)
        passed &= bool(layers) and all(coverage["targets"].values())
        passed &= coverage["categories"] == expected_categories
        passed &= not coverage["unexpected"]
        passed &= all(torch.isfinite(parameter).all().item() for _, parameter in factors)
        return {
            "passed": passed,
            "rank": torch.distributed.get_rank(),
            "coverage": coverage,
            "pipeline_rank": parallel_state.get_pipeline_model_parallel_rank(),
            "tp_rank": parallel_state.get_tensor_model_parallel_rank(),
            "layer_numbers": layers,
            "moe_layer_pattern": list(moe_layers),
            "process_id": os.getpid(),
            "factors": describe_named_tensors(factors),
        }

    def describe_lora_gradients(self):
        gradients = []
        missing = []
        invalid = []
        for name, parameter in get_adapter_parameters(self.actor_module):
            gradient = getattr(parameter, "main_grad", None)
            if gradient is None:
                gradient = parameter.grad
            if gradient is not None:
                if gradient.shape != parameter.shape or not parameter.requires_grad:
                    invalid.append(name)
                gradients.append((name, gradient))
            else:
                missing.append(name)
        finite = bool(gradients) and all(torch.isfinite(value).all().item() for _, value in gradients)
        dtypes = {str(value.dtype) for _, value in gradients}
        return {
            "passed": finite
            and not missing
            and not invalid
            and dtypes == {"torch.float32"}
            and validate_factor_inventory(self.actor_module, self.provider.moe_layer_freq),
            "rank": torch.distributed.get_rank(),
            "gradients": describe_named_tensors(gradients),
            "missing": missing,
            "invalid": invalid,
        }

    def describe_optimizer_state(self):
        state = collect_optimizer_state(self.optimizer)
        moments = list(iter_adam_moments([part["parameter_state"] for part in state]))
        tensors = list(iter_state_tensors(state))
        return {
            "passed": bool(moments) and all(torch.isfinite(value).all().item() for _, value in tensors),
            "rank": torch.distributed.get_rank(),
            "tensor_count": len(tensors),
            "moment_count": len(moments),
            "dtypes": sorted({str(value.dtype) for _, value in tensors}),
        }

    def describe_router_bias(self):
        tensors = [
            (f"chunk{chunk_index}.{name}", tensor)
            for chunk_index, chunk in enumerate(self.actor_module)
            for name, tensor in (tuple(chunk.named_parameters()) + tuple(chunk.named_buffers()))
            if "router" in name and "bias" in name
        ]
        tensors = list(dict(tensors).items())
        if not tensors or not all(torch.isfinite(tensor).all().item() for _, tensor in tensors):
            raise ValueError("Live router-bias evidence is missing or nonfinite")
        return {
            "rank": torch.distributed.get_rank(),
            "update_rate": self.provider.moe_router_bias_update_rate,
            "tensors": describe_named_tensors(tensors),
        }

    def describe_restorable_state(self):
        return {
            "rank": torch.distributed.get_rank(),
            "model": fingerprint_state(dict(get_adapter_parameters(self.actor_module))),
            "optimizer": fingerprint_state(collect_optimizer_state(self.optimizer)),
            "scheduler": fingerprint_state(self.scheduler.state_dict()),
        }

    def mutate_restorable_state(self):
        parameters = get_adapter_parameters(self.actor_module)
        if not parameters:
            raise ValueError("No LoRA parameter is available for restore mutation")
        with torch.no_grad():
            parameters[0][1].view(-1)[0] += 1
        state = collect_optimizer_state(self.optimizer)
        moments = list(iter_adam_moments([part["parameter_state"] for part in state]))
        if not moments:
            raise ValueError("No local Adam moments are available for restore mutation")
        before = fingerprint_state(state)
        with torch.no_grad():
            for moment in moments:
                moment.view(-1)[0] += 1
        if fingerprint_state(collect_optimizer_state(self.optimizer)) == before:
            raise ValueError("Public optimizer state did not mutate live moment buffers")
        self.scheduler.step(increment=1)
        return self.describe_restorable_state()

    def describe_checkpoint(self, checkpoint_dir):
        if torch.distributed.get_rank() != 0:
            return None
        files = sorted(path for path in Path(checkpoint_dir).rglob("*") if path.is_file())
        return {
            "file_count": len(files),
            "total_bytes": sum(path.stat().st_size for path in files),
        }

    def describe_export_stream(self):
        keep_state = self._is_lora_sync_writer_rank()
        exported = {}
        for name, tensor in self.bridge.export_adapter_weights(self.actor_module, cpu=keep_state, show_progress=False):
            if keep_state:
                exported[name] = describe_tensor(tensor)
        passed = not keep_state or (
            bool(exported) and {item["dtype"] for item in exported.values()} == {"torch.float32"}
        )
        return {
            "passed": passed,
            "rank": torch.distributed.get_rank(),
            "writer": keep_state,
            "tensors": exported,
        }

    async def _save_lora_adapters_and_sync(self, lora_sync_path, inference_engine_client, lora_name="skyrl-lora"):
        client = PublicationAuditClient(
            proxy_url=inference_engine_client.proxy_url,
            server_urls=inference_engine_client.server_urls,
            data_parallel_size=inference_engine_client.data_parallel_size,
        )
        try:
            await super()._save_lora_adapters_and_sync(lora_sync_path, client, lora_name)
            self._h1_receiver_loads = client.publication_receipts if torch.distributed.get_rank() == 0 else None
        finally:
            await client.aclose()
        self._h1_publication_index += 1
        self._h1_last_lora_path = lora_sync_path

    def describe_disk_publication(self):
        if self._h1_last_lora_path is None:
            raise RuntimeError("No disk publication has completed")
        weights = Path(self._h1_last_lora_path) / "adapter_model.safetensors"
        config = Path(self._h1_last_lora_path) / "adapter_config.json"
        writer = self._is_lora_sync_writer_rank()
        tensors = {}
        artifact_sha256 = None
        if writer:
            artifact_sha256 = hash_file(weights)
            with safe_open(weights, framework="pt", device="cpu") as adapter:
                tensors = {name: describe_tensor(adapter.get_tensor(name)) for name in adapter.keys()}
            if {item["dtype"] for item in tensors.values()} != {"torch.float32"}:
                raise ValueError("Published LoRA safetensors are not FP32")
            if not config.is_file():
                raise ValueError("Published adapter config is missing")
        return {
            "rank": torch.distributed.get_rank(),
            "writer": writer,
            "publication_index": self._h1_publication_index,
            "artifact_sha256": artifact_sha256,
            "receiver_loads": self._h1_receiver_loads,
            "tensors": tensors,
            "path": str(self._h1_last_lora_path),
            "transport": {
                "merge_lora": self.cfg.policy.megatron_config.lora_config.merge_lora,
                "file_path": self.cfg.policy.model.lora.lora_sync_path,
                "colocate_all": self.cfg.placement.colocate_all,
            },
        }
