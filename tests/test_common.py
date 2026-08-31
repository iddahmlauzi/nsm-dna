import torch

from nsm_dna.models.common import (
    RMSNorm,
    TransformerBlock,
    apply_rope,
    precompute_rope_cosine_and_sine,
)


def test_rms_norm_sets_unit_root_mean_square_without_centering() -> None:
    norm = RMSNorm(normalized_dim=2, eps=0.0)
    x = torch.tensor([[[[1.0, 3.0]]]])

    normalized = norm(x)

    root_mean_square = normalized.square().mean(dim=-1).sqrt()
    torch.testing.assert_close(root_mean_square, torch.ones_like(root_mean_square))
    assert not torch.allclose(normalized.mean(dim=-1), torch.zeros(1, 1, 1))


def test_rope_rotates_nonzero_positions() -> None:
    positions = torch.tensor([0, 1])
    cosine, sine = precompute_rope_cosine_and_sine(
        positions,
        head_dim=4,
        base=10000.0,
    )
    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]]])

    rotated = apply_rope(x, cosine, sine)

    torch.testing.assert_close(rotated[:, :, 0], x[:, :, 0])
    assert not torch.allclose(rotated[:, :, 1], x[:, :, 1])
    torch.testing.assert_close(
        torch.linalg.vector_norm(rotated, dim=-1),
        torch.linalg.vector_norm(x, dim=-1),
    )


def test_transformer_block_applies_rotary_embeddings() -> None:
    block = TransformerBlock(embed_dim=2, num_heads=1, dropout=0.0)
    identity = torch.eye(2)

    with torch.no_grad():
        block.attn.qkv_proj.weight.copy_(torch.cat([identity, identity, identity]))
        block.attn.out_proj.weight.copy_(identity)
        for parameter in block.mlp.parameters():
            parameter.zero_()

    x = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    cosine, sine = precompute_rope_cosine_and_sine(
        torch.tensor([0, 1]),
        head_dim=2,
        base=10000.0,
    )

    output_without_rope = block(x)
    output_with_rope = block(x, rotary_embeddings=(cosine, sine))

    assert not torch.allclose(output_with_rope, output_without_rope)


def test_transformer_block_respects_attention_mask() -> None:
    block = TransformerBlock(embed_dim=2, num_heads=1, dropout=0.0)
    identity = torch.eye(2)

    with torch.no_grad():
        block.attn.qkv_proj.weight.copy_(torch.cat([identity, identity, identity]))
        block.attn.out_proj.weight.copy_(identity)
        for parameter in block.mlp.parameters():
            parameter.zero_()

    x = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    x_with_changed_second_position = torch.tensor(
        [[[1.0, 0.0], [1.0, 0.0]]]
    )
    attention_mask = torch.eye(2, dtype=torch.bool).reshape(1, 1, 2, 2)

    masked_output = block(x, attention_mask=attention_mask)
    masked_output_with_changed_second_position = block(
        x_with_changed_second_position,
        attention_mask=attention_mask,
    )
    unmasked_output = block(x)
    unmasked_output_with_changed_second_position = block(
        x_with_changed_second_position
    )

    torch.testing.assert_close(
        masked_output[:, 0],
        masked_output_with_changed_second_position[:, 0],
    )
    assert not torch.allclose(
        unmasked_output[:, 0],
        unmasked_output_with_changed_second_position[:, 0],
    )
