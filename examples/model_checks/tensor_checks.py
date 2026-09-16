"""Preserve original tensor storage identity alongside logical value hashes."""

import hashlib

import torch


def describe_tensor_storage(tensor):
    return {
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "storage_offset": tensor.storage_offset(),
        "storage_nbytes": tensor.untyped_storage().nbytes(),
        "storage_data_ptr": tensor.untyped_storage().data_ptr(),
    }


def describe_tensor(tensor):
    value = tensor.detach().contiguous()
    payload = value.reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
    return {**describe_tensor_storage(tensor), "sha256": hashlib.sha256(payload).hexdigest()}
