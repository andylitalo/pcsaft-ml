"""Tests for ChemBERTa fine-tuning module (Step 04).

Smoke tests verify that model components instantiate and produce correct
output shapes without crashing.  They do NOT assert high accuracy.
"""

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# ChemBERTaForPCSAFT model tests
# ---------------------------------------------------------------------------


class TestChemBERTaForPCSAFT:
    """Tests for the custom ChemBERTa + regression head model."""

    @pytest.fixture
    def model(self):
        """Load ChemBERTa model (downloads weights on first run)."""
        from model.hf.chemberta_model import ChemBERTaForPCSAFT

        return ChemBERTaForPCSAFT()

    @pytest.fixture
    def tokenizer(self):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

    @pytest.fixture
    def sample_inputs(self, tokenizer):
        """Tokenized SMILES for a small batch."""
        smiles = ["CCO", "c1ccccc1", "CC=O"]
        return tokenizer(
            smiles,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

    def test_forward_pass_shape(self, model, sample_inputs):
        """Forward pass returns predictions of shape (batch, 3)."""
        model.eval()
        with torch.no_grad():
            result = model(
                input_ids=sample_inputs["input_ids"],
                attention_mask=sample_inputs["attention_mask"],
            )
        assert "predictions" in result
        assert result["predictions"].shape == (3, 3)
        assert result["loss"] is None

    def test_forward_with_labels(self, model, sample_inputs):
        """Forward pass with labels returns MSE loss."""
        labels = torch.randn(3, 3)
        result = model(
            input_ids=sample_inputs["input_ids"],
            attention_mask=sample_inputs["attention_mask"],
            labels=labels,
        )
        assert result["loss"] is not None
        assert result["loss"].shape == ()
        assert result["loss"].item() > 0

    def test_mc_dropout_uncertainty(self, model, sample_inputs):
        """MC Dropout returns means and stds for all targets."""
        result = model.predict_with_uncertainty(
            input_ids=sample_inputs["input_ids"],
            attention_mask=sample_inputs["attention_mask"],
            n_forward=5,
        )
        from model.hf.chemberta_model import TARGETS

        for target in TARGETS:
            assert target in result
            assert f"{target}_std" in result
            assert result[target].shape == (3,)
            assert result[f"{target}_std"].shape == (3,)
            assert (result[f"{target}_std"] >= 0).all()

        # After predict_with_uncertainty, model should be in eval mode
        assert not model.training


# ---------------------------------------------------------------------------
# PCSAFTSmilesDataset tests
# ---------------------------------------------------------------------------


class TestPCSAFTSmilesDataset:
    """Tests for the HuggingFace-compatible SMILES dataset."""

    def test_dataset_length(self):
        """Dataset reports correct number of samples."""
        from model.hf.dataset import PCSAFTSmilesDataset

        smiles = ["CCO", "c1ccccc1", "CC=O", "C"]
        targets = torch.randn(4, 3)
        ds = PCSAFTSmilesDataset(smiles, targets=targets)
        assert len(ds) == 4

    def test_dataset_getitem(self):
        """Dataset __getitem__ returns correct keys and shapes."""
        from model.hf.dataset import PCSAFTSmilesDataset

        smiles = ["CCO", "c1ccccc1"]
        targets = torch.randn(2, 3)
        ds = PCSAFTSmilesDataset(smiles, targets=targets, max_length=64)
        item = ds[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item
        assert item["input_ids"].shape == (64,)
        assert item["labels"].shape == (3,)

    def test_dataset_inference_mode(self):
        """Dataset without targets omits labels from items."""
        from model.hf.dataset import PCSAFTSmilesDataset

        smiles = ["CCO"]
        ds = PCSAFTSmilesDataset(smiles, targets=None)
        item = ds[0]
        assert "input_ids" in item
        assert "labels" not in item


# ---------------------------------------------------------------------------
# Registry wrapper tests
# ---------------------------------------------------------------------------


class TestChemBERTaRegistryWrapper:
    """Tests for the ChemBERTa model wrapper in the registry."""

    def test_chemberta_registered(self):
        """ChemBERTa is registered in the model registry."""
        from model.registry import list_models

        assert "chemberta" in list_models()

    def test_chemberta_has_required_methods(self):
        """ChemBERTa wrapper exposes load, predict, predict_with_uncertainty."""
        from model.registry import get_model

        model = get_model("chemberta")
        assert hasattr(model, "load")
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_with_uncertainty")

    def test_chemberta_load_and_predict(self):
        """ChemBERTa wrapper loads saved model and returns correct schema."""
        from pathlib import Path

        from model.data.load import TARGETS
        from model.registry import get_model

        saved_dir = Path(__file__).resolve().parent.parent / "model" / "saved"
        if not (saved_dir / "chemberta" / "chemberta_pcsaft.pt").exists():
            pytest.skip("ChemBERTa not trained; run training first")

        model = get_model("chemberta")
        model.load()
        result = model.predict(["CCO", "c1ccccc1", "CC=O"])

        assert isinstance(result, dict)
        for target in TARGETS:
            assert target in result
            assert isinstance(result[target], np.ndarray)
            assert len(result[target]) == 3
            # Predictions should be denormalized (physical values)
            assert np.isfinite(result[target]).all()


# ---------------------------------------------------------------------------
# Expanded candidate generation tests
# ---------------------------------------------------------------------------


class TestExpandedCandidateGeneration:
    """Tests for the expanded scaffold families in screening/generate.py."""

    def test_hfo_only_backward_compatible(self):
        """HFO-only generation still works and produces ~63 candidates."""
        from screening.generate import generate_hfo_candidates

        candidates = generate_hfo_candidates(scaffold_set="hfo")
        assert len(candidates) > 50
        # Each candidate is (SMILES, Mol) tuple
        assert isinstance(candidates[0], tuple)
        assert len(candidates[0]) == 2

    def test_broad_produces_500_plus(self):
        """Broad scaffold set generates 500+ candidates."""
        from screening.generate import generate_hfo_candidates

        candidates = generate_hfo_candidates(scaffold_set="broad")
        assert len(candidates) >= 500, (
            f"Expected >= 500 candidates with broad set, got {len(candidates)}"
        )

    def test_scaffold_sets_exist(self):
        """All scaffold set keys are defined."""
        from screening.generate import SCAFFOLD_SETS

        assert "hfo" in SCAFFOLD_SETS
        assert "broad" in SCAFFOLD_SETS
        assert "all" in SCAFFOLD_SETS

    def test_all_smiles_are_valid(self):
        """All generated candidates have valid RDKit Mol objects."""
        from rdkit import Chem

        from screening.generate import generate_hfo_candidates

        candidates = generate_hfo_candidates(scaffold_set="broad")
        for smi, mol in candidates:
            assert mol is not None, f"Invalid Mol for SMILES: {smi}"
            assert Chem.MolToSmiles(mol) == smi
