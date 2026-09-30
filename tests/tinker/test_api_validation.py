import base64
from unittest.mock import AsyncMock

import httpx
import pytest
import tinker.types as sdk_types
from pydantic import TypeAdapter, ValidationError

from skyrl.tinker import api, types

_B64_PNG = base64.b64encode(b"\x89PNG").decode()


def _make_datum() -> api.Datum:
    return api.Datum(
        model_input=api.ModelInput(chunks=[api.EncodedTextChunk(tokens=[1, 2, 3])]),
        loss_fn_inputs={
            "target_tokens": api.TensorData(data=[2, 3, 4]),
            "weights": api.TensorData(data=[1.0, 1.0, 1.0]),
        },
    )


def test_forward_backward_input_accepts_ppo_threshold_keys():
    req = api.ForwardBackwardInput(
        data=[_make_datum()],
        loss_fn="ppo",
        loss_fn_config={"clip_low_threshold": 0.9, "clip_high_threshold": 1.1},
    )
    assert req.loss_fn_config == {"clip_low_threshold": 0.9, "clip_high_threshold": 1.1}


def test_forward_backward_input_accepts_ppo_value_clip():
    req = api.ForwardBackwardInput(
        data=[_make_datum()],
        loss_fn="ppo",
        loss_fn_config={"value_clip": 0.2},
    )
    assert req.loss_fn_config == {"value_clip": 0.2}


def test_forward_backward_input_accepts_ppo_critic_value_clip():
    req = api.ForwardBackwardInput(
        data=[_make_datum()],
        loss_fn="ppo_critic",
        loss_fn_config={"value_clip": 0.2},
    )
    assert req.loss_fn_config == {"value_clip": 0.2}


def test_forward_backward_input_accepts_dppo_delta_keys():
    req = api.ForwardBackwardInput(
        data=[_make_datum()],
        loss_fn="dppo",
        loss_fn_config={"delta_low": 0.2, "delta_high": 0.2},
    )
    assert req.loss_fn_config == {"delta_low": 0.2, "delta_high": 0.2}


def test_forward_backward_input_rejects_invalid_dppo_loss_fn_config_keys():
    with pytest.raises(ValidationError, match="Invalid loss_fn_config keys"):
        api.ForwardBackwardInput(
            data=[_make_datum()],
            loss_fn="dppo",
            loss_fn_config={"clip_low_threshold": 0.9},
        )


def test_forward_backward_input_rejects_invalid_ppo_loss_fn_config_keys():
    with pytest.raises(ValidationError, match="Invalid loss_fn_config keys"):
        api.ForwardBackwardInput(
            data=[_make_datum()],
            loss_fn="ppo",
            loss_fn_config={"clip_ratio": 0.2},
        )


def test_forward_backward_input_rejects_loss_fn_config_for_cross_entropy():
    with pytest.raises(ValidationError, match="does not accept loss_fn_config keys"):
        api.ForwardBackwardInput(
            data=[_make_datum()],
            loss_fn="cross_entropy",
            loss_fn_config={"clip_low_threshold": 0.9},
        )


def test_datum_to_types_defaults_values_and_returns_to_empty():
    datum = _make_datum().to_types()
    assert datum.loss_fn_inputs.values.data == []
    assert datum.loss_fn_inputs.returns.data == []


def test_datum_to_types_preserves_values_and_returns():
    datum = api.Datum(
        model_input=api.ModelInput(chunks=[api.EncodedTextChunk(tokens=[1, 2, 3])]),
        loss_fn_inputs={
            "target_tokens": api.TensorData(data=[2, 3, 4]),
            "weights": api.TensorData(data=[1.0, 1.0, 1.0]),
            "values": api.TensorData(data=[0.1, 0.2, 0.3]),
            "returns": api.TensorData(data=[0.4, 0.5, 0.6]),
        },
    ).to_types()

    assert datum.loss_fn_inputs.values.data == [0.1, 0.2, 0.3]
    assert datum.loss_fn_inputs.returns.data == [0.4, 0.5, 0.6]


@pytest.mark.parametrize("optimizer", [False, True])
def test_load_weights_request_preserves_optimizer_choice(optimizer):
    request = api.LoadWeightsRequest(
        model_id="model_test",
        path="tinker://source_model/weights/checkpoint",
        optimizer=optimizer,
    )

    assert request.optimizer is optimizer


# --- ModelInputChunk discriminator tests (api) ---

_api_adapter = TypeAdapter(api.ModelInputChunk)


class TestApiChunkDiscriminatorWithoutType:
    """Chunks resolved when ``type`` is absent (exclude_unset case)."""

    def test_encoded_text(self):
        obj = _api_adapter.validate_python({"tokens": [1, 2]})
        assert isinstance(obj, api.EncodedTextChunk)

    def test_image(self):
        obj = _api_adapter.validate_python({"data": _B64_PNG, "format": "png"})
        assert isinstance(obj, api.ImageChunk)

    def test_image_asset_pointer(self):
        obj = _api_adapter.validate_python({"format": "png", "location": "s3://bucket/img.png"})
        assert isinstance(obj, api.ImageAssetPointerChunk)


def test_api_chunk_discriminator_rejects_ambiguous_payload():
    with pytest.raises(ValueError, match="Ambiguous model chunk type"):
        _api_adapter.validate_python({"tokens": [1, 2], "data": _B64_PNG, "format": "png"})


def test_api_chunk_discriminator_rejects_unrecognised_payload():
    with pytest.raises(ValidationError):
        _api_adapter.validate_python({"format": "png"})


# --- ImageChunk base64 round-trip tests ---

_RAW_PNG = b"\x89PNG\r\n\x1a\nfake_image_data"
_B64_PNG_FULL = base64.b64encode(_RAW_PNG).decode()


class TestImageChunkBase64RoundTrip:
    """Verify that image data survives the api -> types -> JSON -> types cycle."""

    def test_to_types_preserves_bytes(self):
        api_chunk = api.ImageChunk.model_validate({"data": _B64_PNG_FULL, "format": "png"})
        assert api_chunk.data == _RAW_PNG

    def test_json_round_trip(self):

        api_chunk = api.ImageChunk.model_validate({"data": _B64_PNG_FULL, "format": "png"})
        types_chunk = api_chunk.to_types()
        assert types_chunk.data == _RAW_PNG

        json_dict = types_chunk.model_dump(mode="json")
        assert json_dict["data"] == _B64_PNG_FULL

        recovered = types.ImageChunk.model_validate(json_dict)
        assert recovered.data == _RAW_PNG

    def test_nested_in_model_input(self):

        api_chunk = api.ImageChunk.model_validate({"data": _B64_PNG_FULL, "format": "png"})
        model_input = api.ModelInput(chunks=[api_chunk])
        types_input = model_input.to_types()

        json_dict = types_input.model_dump(mode="json")
        recovered = types.ModelInput.model_validate(json_dict)
        assert recovered.chunks[0].data == _RAW_PNG


@pytest.mark.asyncio
@pytest.mark.parametrize("num_samples", [1, 3])
async def test_asample_http_returns_sequence_ids_for_sdk(monkeypatch, num_samples):
    session = AsyncMock()

    async def get_session():
        yield session

    monkeypatch.setitem(api.app.dependency_overrides, api.get_session, get_session)
    monkeypatch.setattr(api.app.state, "external_future_store", None, raising=False)
    monkeypatch.setattr(api, "create_future", AsyncMock(return_value=123))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as client:
        response = await client.post(
            "/api/v1/asample",
            json={
                "base_model": "test-model",
                "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1, 2]}]},
                "sampling_params": {"max_tokens": 1},
                "num_samples": num_samples,
            },
        )
    assert response.status_code == 200
    body = response.json()
    ids = body["sample_sequence_ids"]
    assert len(ids) == num_samples
    assert len(set(ids)) == num_samples
    promise = sdk_types.UntypedAPIFuture.model_validate(body)
    assert promise.request_id == "123"
    # SDK 0.25 ignores this field; 0.27 requires it when resolving samples.
    if "sample_sequence_ids" in sdk_types.UntypedAPIFuture.model_fields:
        assert promise.sample_sequence_ids == ids


def test_non_sampling_future_omits_sample_sequence_ids():
    response = api.FutureResponse(future_id="123", request_id="123")
    assert response.model_dump() == {"future_id": "123", "request_id": "123", "status": "pending"}
