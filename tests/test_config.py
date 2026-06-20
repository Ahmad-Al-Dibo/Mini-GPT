import pytest

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.miniGPT import create_config

def test_create_default_config():
    cfg = create_config()

    assert cfg.embed_dim == 64
    assert cfg.num_blocks == 2
    assert cfg.block_size == 8
    assert cfg.batch_size == 64


def test_override_single_parameter():
    cfg = create_config(embed_dim=256)

    assert cfg.embed_dim == 256


def test_override_multiple_parameters():
    cfg = create_config(
        embed_dim=128,
        num_blocks=4,
        learning_rate=1e-4,
    )

    assert cfg.embed_dim == 128
    assert cfg.num_blocks == 4
    assert cfg.learning_rate == 1e-4


def test_override_tokenizer_settings():
    cfg = create_config(
        tokenizer_type="sentencepiece",
        max_vocab=16000,
    )

    assert cfg.tokenizer_type == "sentencepiece"
    assert cfg.max_vocab == 16000


def test_unknown_parameter_raises():
    with pytest.raises(ValueError, match="Unknown configuration"):
        create_config(non_existing_setting=123)


def test_defaults_are_preserved():
    cfg = create_config(embed_dim=128)

    assert cfg.embed_dim == 128
    assert cfg.batch_size == 64
    assert cfg.learning_rate == 1e-3
    assert cfg.max_vocab == 50000


def test_long_context_override():
    cfg = create_config(
        run_long_context_evaluation=True,
        long_context_block_sizes=[64, 128, 256],
    )

    assert cfg.run_long_context_evaluation is True
    assert cfg.long_context_block_sizes == [64, 128, 256]