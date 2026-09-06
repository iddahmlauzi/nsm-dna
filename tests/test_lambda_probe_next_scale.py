import torch
import torch.nn as nn

from scripts.evaluation.lambda_probe_next_scale import (
    NSMWindowEncoder,
    evaluate_representations,
    segment_window_starts,
)


def test_window_encoder_preserves_prefix_and_scale_features() -> None:
    class StubTokenizer(nn.Module):
        context_length = 2
        scale_lengths = [1, 2]

        def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
            return token_ids.unsqueeze(-1).float()

        def encode_indices(self, token_ids: torch.Tensor) -> list[torch.Tensor]:
            return [token_ids[:, :1], token_ids]

        def indices_to_next_scale_inputs(
            self,
            indices_by_scale: list[torch.Tensor],
        ) -> list[torch.Tensor]:
            return [indices_by_scale[0].expand(-1, 2).unsqueeze(-1).float()]

    class StubModel(nn.Module):
        def encode(
            self,
            scale_inputs: list[torch.Tensor],
            *,
            prefix: torch.Tensor,
        ) -> torch.Tensor:
            return torch.tensor(
                [
                    [
                        [1.0, 2.0],
                        [3.0, 4.0],
                        [5.0, 6.0],
                        [7.0, 8.0],
                    ]
                ]
            )

    encoder = NSMWindowEncoder(StubModel(), StubTokenizer())

    embedding = encoder(torch.tensor([[0, 1, 2, 3]]))

    # The prefix and the section predicting scale 2 are pooled separately.
    torch.testing.assert_close(
        embedding,
        torch.tensor([[2.0, 3.0, 6.0, 7.0]]),
    )


def test_window_encoder_supports_a_complete_target_without_prefix() -> None:
    class StubTokenizer(nn.Module):
        context_length = 2
        scale_lengths = [1, 2]

        def encode_indices(self, token_ids: torch.Tensor) -> list[torch.Tensor]:
            return [token_ids[:, :1], token_ids]

        def indices_to_next_scale_inputs(
            self,
            indices_by_scale: list[torch.Tensor],
        ) -> list[torch.Tensor]:
            return [indices_by_scale[0].expand(-1, 2).unsqueeze(-1).float()]

    class StubModel(nn.Module):
        def encode(
            self,
            scale_inputs: list[torch.Tensor],
            *,
            prefix: torch.Tensor | None,
        ) -> torch.Tensor:
            assert prefix is None
            return torch.tensor([[[5.0, 6.0], [7.0, 8.0]]])

    encoder = NSMWindowEncoder(StubModel(), StubTokenizer())

    embedding = encoder(torch.tensor([[2, 3]]))

    torch.testing.assert_close(embedding, torch.tensor([[6.0, 7.0]]))


def test_segment_windows_advance_by_one_target_block() -> None:
    assert segment_window_starts(2_000, 256, 128) == [
        *range(0, 1_665, 128),
        1_744,
    ]


def test_segment_windows_do_not_duplicate_an_aligned_final_window() -> None:
    assert segment_window_starts(512, 256, 128) == [0, 128, 256]


def test_evaluate_representations_splits_prefix_and_each_scale(monkeypatch) -> None:
    combined_embeddings = {
        "train": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]).numpy(),
        "test": torch.tensor([[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]]).numpy(),
    }
    observed_embeddings = {}

    def stub_extract(*args, description: str, **kwargs):
        split_name = description.rsplit(" ", maxsplit=1)[-1]
        return combined_embeddings[split_name]

    def stub_evaluate(name, embeddings, *args, **kwargs):
        observed_embeddings[name] = embeddings
        return {"linear_probe": {"mcc": 0.0}}

    monkeypatch.setattr(
        "scripts.evaluation.lambda_probe_next_scale.extract_segment_embeddings",
        stub_extract,
    )
    monkeypatch.setattr(
        "scripts.evaluation.lambda_probe_next_scale.evaluate_probes",
        stub_evaluate,
    )
    config = type(
        "Config",
        (),
        {
            "evaluate_three_layer_probe": False,
            "embedding_batch_size": 1,
            "representation_names": ["prefix", "scale_length_2"],
        },
    )()

    evaluate_representations(
        "trained",
        nn.Identity(),
        model_dim=2,
        scale_lengths=[1, 2],
        window_length=4,
        stride=2,
        splits={
            "train": type("Split", (), {"sequences": []})(),
            "test": type("Split", (), {"sequences": []})(),
        },
        config=config,
        output_directory=None,
        device=torch.device("cpu"),
        include_prefix=True,
    )

    expected = {
        "trained_prefix": [1.0, 2.0],
        "trained_scale_length_2": [5.0, 6.0],
    }
    assert set(observed_embeddings) == set(expected)
    for name, expected_values in expected.items():
        torch.testing.assert_close(
            torch.from_numpy(observed_embeddings[name]["train"]),
            torch.tensor([expected_values]),
        )


def test_evaluate_representations_omits_prefix_for_target_only_windows(
    monkeypatch,
) -> None:
    combined_embeddings = {
        "train": torch.tensor([[1.0, 2.0, 3.0, 4.0]]).numpy(),
        "test": torch.tensor([[5.0, 6.0, 7.0, 8.0]]).numpy(),
    }
    observed_names = []

    def stub_extract(*args, description: str, **kwargs):
        split_name = description.rsplit(" ", maxsplit=1)[-1]
        return combined_embeddings[split_name]

    def stub_evaluate(name, *args, **kwargs):
        observed_names.append(name)
        return {"linear_probe": {"mcc": 0.0}}

    monkeypatch.setattr(
        "scripts.evaluation.lambda_probe_next_scale.extract_segment_embeddings",
        stub_extract,
    )
    monkeypatch.setattr(
        "scripts.evaluation.lambda_probe_next_scale.evaluate_probes",
        stub_evaluate,
    )
    config = type(
        "Config",
        (),
        {
            "evaluate_three_layer_probe": False,
            "embedding_batch_size": 1,
            "representation_names": None,
        },
    )()

    evaluate_representations(
        "trained",
        nn.Identity(),
        model_dim=2,
        scale_lengths=[1, 2],
        window_length=2,
        stride=2,
        splits={
            "train": type("Split", (), {"sequences": []})(),
            "test": type("Split", (), {"sequences": []})(),
        },
        config=config,
        output_directory=None,
        device=torch.device("cpu"),
        include_prefix=False,
    )

    assert observed_names == ["trained_scale_length_1", "trained_scale_length_2"]
