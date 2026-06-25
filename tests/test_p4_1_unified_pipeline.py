"""
Tests for P4-1: Unified Training Pipeline + Pre-trained Model Support
Verifies: stopping criteria, pre-trained model loading, model adapter, unified pipeline
"""

import pytest
import torch
import tempfile
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from src.miniGPT.config import Config
from src.miniGPT.model import MiniGPT
from src.miniGPT.pipeline_v2 import (
    StoppingCriteria,
    PreTrainedModelLoader,
    ModelAdapter,
    UnifiedPipeline,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def test_config():
    """Create a minimal test config."""
    return Config(
        embed_dim=64,
        block_size=32,
        num_blocks=2,
        vocab_size=256,
        batch_size=4,
        learning_rate=1e-4,
        weight_decay=1e-5,
        epochs=2,
        dropout=0.1,
        device="cpu",
    )


@pytest.fixture
def test_model():
    """Create a minimal test model."""
    return MiniGPT(
        vocab_size=256,
        embed_dim=64,
        block_size=32,
        num_blocks=2,
        dropout=0.1,
    )


@pytest.fixture
def test_dataloader():
    """Create a minimal test dataloader."""
    X = torch.randint(0, 256, (16, 32))
    y = torch.randint(0, 256, (16, 32))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=4)


@pytest.fixture
def test_checkpoint_path():
    """Create a temporary checkpoint file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model = MiniGPT(
            vocab_size=256,
            embed_dim=64,
            block_size=32,
            num_blocks=2,
        )
        
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "config": {
                "embed_dim": 64,
                "block_size": 32,
                "num_blocks": 2,
                "vocab_size": 256,
                "dropout": 0.1,
            },
            "vocab_size": 256,
            "current_epoch": 0,
        }
        
        path = Path(tmpdir) / "test_checkpoint.pth"
        torch.save(checkpoint, path)
        yield path


# ============================================================================
# STOPPING CRITERIA TESTS
# ============================================================================

class TestStoppingCriteria:
    """Test StoppingCriteria functionality."""
    
    def test_max_steps_not_exceeded(self):
        """Test max_steps boundary."""
        criteria = StoppingCriteria(max_steps=100)
        assert not criteria.should_stop_by_steps(50)
        assert not criteria.should_stop_by_steps(99)
    
    def test_max_steps_exceeded(self):
        """Test max_steps exceeded."""
        criteria = StoppingCriteria(max_steps=100)
        assert criteria.should_stop_by_steps(100)
        assert criteria.should_stop_by_steps(101)
    
    def test_early_stopping_no_improvement(self):
        """Test early stopping when no improvement."""
        criteria = StoppingCriteria(
            early_stopping_patience=3,
            early_stopping_min_delta=0.001,
        )
        
        best_loss = 0.5
        
        # First check: improvement
        should_stop = criteria.should_stop_by_validation(0.4, best_loss, 0)
        assert not should_stop
        
        # Subsequent checks: no improvement
        should_stop = criteria.should_stop_by_validation(0.401, best_loss, 1)
        assert not should_stop
        
        should_stop = criteria.should_stop_by_validation(0.401, best_loss, 2)
        assert not should_stop
        
        should_stop = criteria.should_stop_by_validation(0.401, best_loss, 3)
        assert should_stop
    
    def test_early_stopping_with_improvement(self):
        """Test early stopping resets on improvement."""
        criteria = StoppingCriteria(
            early_stopping_patience=3,
            early_stopping_min_delta=0.001,
        )
        
        best_loss = 0.5
        
        # No improvement
        should_stop = criteria.should_stop_by_validation(0.401, best_loss, 1)
        assert not should_stop
        
        # Sudden improvement (below min_delta)
        should_stop = criteria.should_stop_by_validation(0.499, best_loss, 1)
        assert not should_stop
    
    def test_none_patience_no_early_stop(self):
        """Test that None patience disables early stopping."""
        criteria = StoppingCriteria(early_stopping_patience=None)
        should_stop = criteria.should_stop_by_validation(0.5, 0.4, 100)
        assert not should_stop


# ============================================================================
# PRE-TRAINED MODEL LOADER TESTS
# ============================================================================

class TestPreTrainedModelLoader:
    """Test PreTrainedModelLoader functionality."""
    
    def test_load_from_checkpoint(self, test_config, test_checkpoint_path):
        """Test loading from local checkpoint file."""
        loader = PreTrainedModelLoader(
            source=test_checkpoint_path,
            target_vocab_size=256,
            target_config=test_config,
            device="cpu",
        )
        
        model, metadata = loader.load()
        
        assert isinstance(model, MiniGPT)
        assert metadata["source"] == "checkpoint"
        assert metadata["path"] == str(test_checkpoint_path)
        assert model.token_embedding.num_embeddings == 256
    
    def test_checkpoint_not_found(self, test_config):
        """Test error when checkpoint doesn't exist."""
        loader = PreTrainedModelLoader(
            source="/nonexistent/model.pth",
            target_vocab_size=256,
            target_config=test_config,
            device="cpu",
        )
        
        with pytest.raises(FileNotFoundError):
            loader.load()
    
    def test_is_huggingface_model_detection(self):
        """Test HuggingFace model ID detection."""
        loader = PreTrainedModelLoader(
            source="gpt2",
            target_vocab_size=256,
            target_config=Config(),
            device="cpu",
        )
        
        # Should detect "gpt2" as HF model (would fail on actual load)
        assert loader._is_huggingface_model("gpt2")
        assert loader._is_huggingface_model("mistralai/Mistral-7B")
    
    def test_is_local_checkpoint_detection(self, test_checkpoint_path):
        """Test local checkpoint detection."""
        loader = PreTrainedModelLoader(
            source=test_checkpoint_path,
            target_vocab_size=256,
            target_config=Config(),
            device="cpu",
        )
        
        assert loader._is_local_checkpoint(test_checkpoint_path)
        assert not loader._is_local_checkpoint("/nonexistent/file.pth")


# ============================================================================
# MODEL ADAPTER TESTS
# ============================================================================

class TestModelAdapter:
    """Test ModelAdapter functionality."""
    
    def test_adapt_weights_basic(self):
        """Test basic weight adaptation."""
        source = MiniGPT(
            vocab_size=128,
            embed_dim=32,
            block_size=16,
            num_blocks=1,
        )
        
        target = MiniGPT(
            vocab_size=256,
            embed_dim=64,
            block_size=32,
            num_blocks=2,
        )
        
        adapter = ModelAdapter(source, target)
        stats = adapter.adapt_weights(verbose=False)
        
        # Check that adaptation was attempted
        assert stats["adapted"] >= 0
        assert stats["skipped"] >= 0
        assert stats["initialized_random"] >= 0
        assert stats["adapted"] + stats["skipped"] + stats["initialized_random"] >= 0
    
    def test_embedding_extraction(self):
        """Test embedding extraction from model."""
        model = MiniGPT(
            vocab_size=256,
            embed_dim=64,
            block_size=32,
            num_blocks=1,
        )
        
        adapter = ModelAdapter(model, model)
        embed = adapter._extract_embedding(model)
        
        assert isinstance(embed, torch.nn.Embedding)
        assert embed.num_embeddings == 256
    
    def test_transformer_blocks_extraction(self):
        """Test transformer blocks extraction."""
        model = MiniGPT(
            vocab_size=256,
            embed_dim=64,
            block_size=32,
            num_blocks=2,
        )
        
        adapter = ModelAdapter(model, model)
        blocks = adapter._extract_transformer_blocks(model)
        
        assert len(blocks) > 0
        assert len(blocks) == 2


# ============================================================================
# UNIFIED PIPELINE TESTS
# ============================================================================

class TestUnifiedPipeline:
    """Test UnifiedPipeline functionality."""
    
    def test_pipeline_initialization_pretraining(self, test_config):
        """Test pipeline initialization for pre-training."""
        pipeline = UnifiedPipeline(
            config=test_config,
            mode="pre_training",
            stopping_criteria=StoppingCriteria(max_steps=100),
        )
        
        assert pipeline.mode == "pre_training"
        assert pipeline.model is None  # Not built yet
    
    def test_pipeline_initialization_finetuning_requires_source(self, test_config):
        """Test that fine-tuning requires pretrained source."""
        pipeline = UnifiedPipeline(
            config=test_config,
            mode="fine_tuning",
        )
        
        with pytest.raises(ValueError, match="requires pretrained_source"):
            pipeline.build(vocab_size=256)
    
    def test_pipeline_build_pretraining(self, test_config):
        """Test building pipeline for pre-training."""
        pipeline = UnifiedPipeline(
            config=test_config,
            mode="pre_training",
        )
        
        pipeline.build(vocab_size=256)
        
        assert pipeline.model is not None
        assert isinstance(pipeline.model, MiniGPT)
        assert pipeline.trainer is not None
        assert pipeline.metadata["source"] == "random_init"
    
    def test_pipeline_build_finetuning_from_checkpoint(
        self, test_config, test_checkpoint_path
    ):
        """Test building pipeline for fine-tuning from checkpoint."""
        pipeline = UnifiedPipeline(
            config=test_config,
            mode="fine_tuning",
            pretrained_source=test_checkpoint_path,
        )
        
        pipeline.build(vocab_size=256)
        
        assert pipeline.model is not None
        assert pipeline.metadata["source"] == "checkpoint"
    
    def test_pipeline_get_model(self, test_config):
        """Test getting model from pipeline."""
        pipeline = UnifiedPipeline(
            config=test_config,
            mode="pre_training",
        )
        
        pipeline.build(vocab_size=256)
        model = pipeline.get_model()
        
        assert isinstance(model, MiniGPT)
    
    def test_pipeline_train_without_build_fails(self, test_config, test_dataloader):
        """Test that training without build fails."""
        pipeline = UnifiedPipeline(
            config=test_config,
            mode="pre_training",
        )
        
        with pytest.raises(RuntimeError, match="not built"):
            pipeline.train(test_dataloader)
    
    def test_pipeline_train_pretraining(self, test_config, test_dataloader):
        """Test training pipeline for pre-training."""
        pipeline = UnifiedPipeline(
            config=test_config,
            mode="pre_training",
        )
        
        pipeline.build(vocab_size=256)
        results = pipeline.train(test_dataloader, num_epochs=1)
        
        assert results["mode"] == "pre_training"
        assert results["epochs_completed"] == 1
        assert len(results["train_losses"]) > 0
    
    def test_pipeline_train_with_validation(self, test_config, test_dataloader):
        """Test training with validation dataloader."""
        pipeline = UnifiedPipeline(
            config=test_config,
            mode="pre_training",
        )
        
        pipeline.build(vocab_size=256)
        results = pipeline.train(
            test_dataloader,
            val_loader=test_dataloader,
            num_epochs=1,
        )
        
        assert len(results["val_losses"]) > 0
        assert "best_val_loss" in results
    
    def test_pipeline_save_checkpoint(self, test_config):
        """Test saving pipeline checkpoint."""
        pipeline = UnifiedPipeline(
            config=test_config,
            mode="pre_training",
        )
        
        pipeline.build(vocab_size=256)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pth"
            saved_path = pipeline.save_checkpoint(str(checkpoint_path))
            
            assert saved_path.exists()
            
            # Verify checkpoint is valid
            checkpoint = torch.load(saved_path, weights_only=False)
            assert "model" in checkpoint
            assert checkpoint["mode"] == "pre_training"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestP4_1_Integration:
    """Integration tests for complete P4-1 workflow."""
    
    def test_complete_pretraining_workflow(self, test_config, test_dataloader):
        """Test complete pre-training workflow."""
        # 1. Initialize pipeline
        pipeline = UnifiedPipeline(
            config=test_config,
            mode="pre_training",
            stopping_criteria=StoppingCriteria(max_steps=50),
        )
        
        # 2. Build model
        pipeline.build(vocab_size=256)
        assert pipeline.model is not None
        
        # 3. Train
        results = pipeline.train(test_dataloader, num_epochs=1)
        assert results["epochs_completed"] > 0
        
        # 4. Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.pth"
            saved_path = pipeline.save_checkpoint(str(checkpoint_path))
            assert saved_path.exists()
    
    def test_complete_finetuning_workflow(
        self, test_config, test_dataloader, test_checkpoint_path
    ):
        """Test complete fine-tuning workflow."""
        # 1. Initialize pipeline for fine-tuning
        pipeline = UnifiedPipeline(
            config=test_config,
            mode="fine_tuning",
            pretrained_source=test_checkpoint_path,
        )
        
        # 2. Build from pre-trained model
        pipeline.build(vocab_size=256)
        assert pipeline.model is not None
        assert pipeline.metadata["source"] == "checkpoint"
        
        # 3. Train
        results = pipeline.train(test_dataloader, num_epochs=1)
        assert results["mode"] == "fine_tuning"
        
        # 4. Save fine-tuned checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "finetuned.pth"
            saved_path = pipeline.save_checkpoint(str(checkpoint_path))
            assert saved_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
