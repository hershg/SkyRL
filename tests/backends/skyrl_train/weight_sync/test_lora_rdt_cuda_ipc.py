import multiprocessing
import pickle

import pytest
import torch

from skyrl.backends.skyrl_train.weight_sync.lora_rdt import producer as producer_module
from skyrl.backends.skyrl_train.weight_sync.lora_rdt.contracts import (
    LoRAAdapterLayout,
    LoRATensorSlice,
    LoRAUpdateRequest,
)
from skyrl.backends.skyrl_train.weight_sync.lora_rdt.producer import LoRardtProducer
from skyrl.backends.skyrl_train.weight_sync.lora_rdt.publication import (
    export_lora_cuda_ipc,
)


def _read_sources_in_sidecar(connection, layout):
    try:
        producer = LoRardtProducer(0, layout)
        for generation in range(2):
            handles = connection.recv()
            request = LoRAUpdateRequest.from_layout(layout, generation)
            producer.publish_cuda(request, handles, 1)
            pulled = producer.pull(generation, ["a", "b"])
            destination = {
                name: value.to(torch.bfloat16) for name, value in pulled.items()
            }
            values = {name: value.cpu().tolist() for name, value in pulled.items()}
            independent = all(
                destination[name].untyped_storage().data_ptr()
                != value.untyped_storage().data_ptr()
                for name, value in pulled.items()
            )
            for value in destination.values():
                value.zero_()
            unchanged = all(
                value.cpu().tolist() == values[name] for name, value in pulled.items()
            )
            del pulled, destination
            released = producer.acknowledge(generation, 0)
            connection.send(
                (
                    values,
                    independent,
                    unchanged,
                    released,
                    producer.retained_generations(),
                )
            )
    finally:
        connection.close()


def test_ipc_export_rejects_cpu_source_before_serialization():
    with pytest.raises(ValueError, match="CUDA float32"):
        export_lora_cuda_ipc({"a": torch.ones(2)})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA IPC")
def test_cuda_ipc_preserves_fp32_values_offsets_and_independent_bf16_buffers():
    layout = LoRAAdapterLayout(
        adapter_name="adapter",
        tensors=(
            LoRATensorSlice("a", (4,), 0, 0, 0, 0, 16),
            LoRATensorSlice("b", (2, 2), 0, 0, 0, 16, 16),
        ),
        source_dtype="float32",
    )
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_read_sources_in_sidecar, args=(child, layout))
    process.start()
    child.close()
    try:
        for generation in range(2):
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                source = (
                    torch.arange(12, dtype=torch.float32, device="cuda")
                    + generation * 16
                    + 0.03125
                )
                tensors = {"a": source[2:6], "b": source[6:10].reshape(2, 2)}
                handles = export_lora_cuda_ipc(tensors)
            assert len(pickle.dumps(handles)) < 4096
            parent.send(handles)
            assert parent.poll(60)
            values, independent, unchanged, released, retained = parent.recv()
            assert values == {
                name: value.cpu().tolist() for name, value in tensors.items()
            }
            assert independent and unchanged and released
            assert retained == []
        process.join(30)
        assert process.exitcode == 0
    finally:
        parent.close()
        if process.is_alive():
            process.terminate()
            process.join(10)


def test_delayed_discarded_publication_consumes_ipc_references_without_retaining(
    monkeypatch,
):
    layout = LoRAAdapterLayout(
        "adapter", "float32", (LoRATensorSlice("a", (2,), 0, 0, 0, 0, 8),)
    )
    producer = LoRardtProducer(0, layout)
    producer.discard(3)
    rebuilt = []

    def rebuild(*arguments):
        rebuilt.append(arguments)
        return torch.ones(2)

    monkeypatch.setattr(producer_module, "rebuild_cuda_tensor", rebuild)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    with pytest.raises(ValueError, match="stale"):
        producer.publish_cuda(
            LoRAUpdateRequest.from_layout(layout, 3), {"a": (None,) * 6 + (7,)}, 1
        )

    assert len(rebuilt) == 1
    assert rebuilt[0][6] == 0
    assert producer.retained_generations() == []
