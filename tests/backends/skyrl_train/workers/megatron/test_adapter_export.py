import tempfile

import torch

from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import (
    _compact_adapter_state,
)


def test_compact_adapter_state_preserves_exact_tensors_and_aliases():
    repeated = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    state = {
        "adapter_a": repeated.clone(),
        "adapter_b": repeated.clone(),
        "adapter_c": repeated.clone(),
        "adapter_d": repeated.clone(),
        "different_shape": repeated.reshape(4, 3).clone(),
        "positive_zero": torch.tensor([0.0]),
        "negative_zero": torch.tensor([-0.0]),
        "nan_a": torch.tensor([float("nan")]),
        "nan_b": torch.tensor([float("nan")]),
    }

    compact = _compact_adapter_state(state)

    assert compact is not None
    assert compact["adapter_a"].data_ptr() == compact["adapter_b"].data_ptr()
    assert compact["nan_a"].data_ptr() == compact["nan_b"].data_ptr()
    assert compact["positive_zero"].data_ptr() != compact["negative_zero"].data_ptr()
    with tempfile.NamedTemporaryFile() as output:
        torch.save(compact, output.name)
        restored = torch.load(output.name, weights_only=True)
    assert restored["adapter_a"].data_ptr() == restored["adapter_b"].data_ptr()
    for name, tensor in state.items():
        assert restored[name].dtype == tensor.dtype
        assert restored[name].shape == tensor.shape
        assert torch.equal(restored[name].view(torch.uint8), tensor.view(torch.uint8))
    converted = {name: tensor.to(torch.bfloat16) for name, tensor in restored.items()}
    assert converted["adapter_a"].data_ptr() != converted["adapter_b"].data_ptr()


def test_compact_adapter_state_keeps_nonduplicated_state_in_safetensors_path():
    state = {
        "first": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "second": torch.arange(12, 24, dtype=torch.float32).reshape(3, 4),
    }

    assert _compact_adapter_state(state) is None
