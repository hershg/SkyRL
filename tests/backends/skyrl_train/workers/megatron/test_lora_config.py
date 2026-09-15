from types import SimpleNamespace
from unittest.mock import patch

from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import MegatronWorker
from skyrl.train.config import SkyRLLoraConfig


def test_configure_lora_passes_a2a_experimental_to_bridge() -> None:
    worker = MegatronWorker.__new__(MegatronWorker)
    worker.cfg = SimpleNamespace(
        bf16=True,
        policy=SimpleNamespace(
            megatron_config=SimpleNamespace(
                lora_config=SimpleNamespace(normalize_moe_lora=False),
            ),
        ),
    )
    worker.provider = SimpleNamespace(num_moe_experts=None)
    lora_config = SkyRLLoraConfig(
        rank=32,
        alpha=32,
        target_modules=["linear_proj"],
        a2a_experimental=True,
    )

    with patch(
        "skyrl.backends.skyrl_train.workers.megatron.megatron_worker.LoRA"
    ) as bridge_lora:
        worker.configure_lora(lora_config)

    assert worker.lora_cls is bridge_lora.return_value
    assert bridge_lora.call_args.kwargs["a2a_experimental"] is True
