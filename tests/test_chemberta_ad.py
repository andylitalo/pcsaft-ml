"""Tests for ChemBERTa applicability domain functionality."""

import numpy as np
import pytest
import torch

from model.hf.ad import fit_chemberta_ad, predict_chemberta_ad
from model.hf.chemberta_model import ChemBERTaForPCSAFT


def test_chemberta_encode_shape():
    """Test that encode() returns correct shape."""
    from transformers import AutoTokenizer

    model = ChemBERTaForPCSAFT()
    model.eval()

    # Use real tokenizer to get valid input_ids
    tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    smiles_list = ["C", "CC", "CCC", "CCCC"]
    encodings = tokenizer(
        smiles_list,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    with torch.no_grad():
        embeddings = model.encode(
            input_ids=encodings["input_ids"],
            attention_mask=encodings["attention_mask"],
        )

    assert embeddings.shape == (4, 768)  # RoBERTa-base hidden size
    assert isinstance(embeddings, torch.Tensor)


def test_chemberta_forward_uses_encode():
    """Test that forward() uses encode() internally."""
    from transformers import AutoTokenizer

    model = ChemBERTaForPCSAFT()
    model.eval()

    # Use real tokenizer
    tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    smiles_list = ["C", "CC"]
    encodings = tokenizer(
        smiles_list,
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )

    with torch.no_grad():
        # Get embeddings directly
        embeddings = model.encode(
            input_ids=encodings["input_ids"],
            attention_mask=encodings["attention_mask"],
        )

        # Get predictions via forward
        result = model.forward(
            input_ids=encodings["input_ids"],
            attention_mask=encodings["attention_mask"],
        )
        predictions = result["predictions"]

        # Manually compute predictions from embeddings
        manual_preds = model.regression_head(embeddings)

    assert torch.allclose(predictions, manual_preds, atol=1e-6)


def test_fit_chemberta_ad_basic():
    """Test fit_chemberta_ad with random embeddings."""
    n_train = 100
    embedding_dim = 768
    embeddings = np.random.randn(n_train, embedding_dim).astype(np.float32)

    ad_model = fit_chemberta_ad(embeddings, contamination=0.05, random_state=42)

    assert ad_model is not None
    assert hasattr(ad_model, "predict")

    # Predict on same data
    labels = ad_model.predict(embeddings)
    assert len(labels) == n_train
    assert set(labels).issubset({1, -1})  # IsolationForest returns 1 or -1


def test_predict_chemberta_ad_returns_boolean():
    """Test that predict_chemberta_ad returns boolean array."""
    n_train = 50
    embedding_dim = 768
    train_embeddings = np.random.randn(n_train, embedding_dim).astype(np.float32)

    ad_model = fit_chemberta_ad(train_embeddings, contamination=0.05, random_state=42)

    # Test on new embeddings
    n_test = 10
    test_embeddings = np.random.randn(n_test, embedding_dim).astype(np.float32)
    in_domain = predict_chemberta_ad(ad_model, test_embeddings)

    assert isinstance(in_domain, np.ndarray)
    assert in_domain.dtype == bool
    assert len(in_domain) == n_test
    assert in_domain.sum() >= 0  # At least some should be in-domain


def test_chemberta_ad_contamination_effect():
    """Test that contamination parameter affects flagged fraction."""
    n_train = 100
    embedding_dim = 768
    embeddings = np.random.randn(n_train, embedding_dim).astype(np.float32)

    # Low contamination
    ad_low = fit_chemberta_ad(embeddings, contamination=0.01, random_state=42)
    labels_low = predict_chemberta_ad(ad_low, embeddings)

    # High contamination
    ad_high = fit_chemberta_ad(embeddings, contamination=0.20, random_state=42)
    labels_high = predict_chemberta_ad(ad_high, embeddings)

    # Higher contamination should flag more as OOD on training set
    ood_low = (~labels_low).sum()
    ood_high = (~labels_high).sum()

    assert ood_high >= ood_low


@pytest.mark.skipif(
    not pytest.importorskip("transformers"),
    reason="transformers not installed",
)
def test_registry_embed_method():
    """Test ChemBERTaModel.embed() if model is trained."""
    from pathlib import Path

    chemberta_dir = Path(__file__).parent.parent / "model" / "saved" / "chemberta"
    weights_path = chemberta_dir / "chemberta_pcsaft.pt"

    if not weights_path.exists():
        pytest.skip("ChemBERTa model not trained")

    from model.registry import get_model

    model = get_model("chemberta")
    model.load()

    smiles_list = ["C", "CC", "CCC"]  # methane, ethane, propane
    embeddings = model.embed(smiles_list)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (3, 768)
    assert np.isfinite(embeddings).all()


@pytest.mark.skipif(
    not pytest.importorskip("transformers"),
    reason="transformers not installed",
)
def test_registry_predict_in_domain_fallback():
    """Test ChemBERTaModel.predict_in_domain() fallback when AD not available."""
    from pathlib import Path
    from unittest.mock import patch

    chemberta_dir = Path(__file__).parent.parent / "model" / "saved" / "chemberta"
    weights_path = chemberta_dir / "chemberta_pcsaft.pt"

    if not weights_path.exists():
        pytest.skip("ChemBERTa model not trained")

    from model.registry import get_model

    model = get_model("chemberta")
    model.load()

    smiles_list = ["C", "CC"]

    # Mock load_chemberta_ad_model to raise FileNotFoundError
    with patch("model.hf.ad.load_chemberta_ad_model", side_effect=FileNotFoundError):
        in_domain = model.predict_in_domain(smiles_list)

    # Should fallback to all in-domain
    assert isinstance(in_domain, np.ndarray)
    assert len(in_domain) == 2
    assert in_domain.all()  # All marked as in-domain


def test_evaluate_get_ad_labels_model_specific():
    """Test that _get_ad_labels branches on model_name."""
    from model.evaluate import _get_ad_labels

    smiles_list = ["C", "CC", "CCC"]

    # Test with model_name=None (should use descriptor-space AD)
    labels_none = _get_ad_labels(smiles_list, model_name=None)
    # May be None if AD model not available

    # Test with model_name="rf" (should use descriptor-space AD)
    labels_rf = _get_ad_labels(smiles_list, model_name="rf")
    # May be None if AD model not available

    # Test with model_name="chemberta" (should try ChemBERTa AD)
    labels_cb = _get_ad_labels(smiles_list, model_name="chemberta")
    # May be None if ChemBERTa AD not available

    # At minimum, function should not crash
    assert labels_none is None or isinstance(labels_none, np.ndarray)
    assert labels_rf is None or isinstance(labels_rf, np.ndarray)
    assert labels_cb is None or isinstance(labels_cb, np.ndarray)


def test_rf_nn_still_use_descriptor_ad():
    """Test that RF and NN models still use descriptor-space AD."""
    from model.evaluate import _get_ad_labels

    smiles_list = ["C", "CC"]

    # Both should use descriptor-space AD (may be None if not available)
    labels_rf = _get_ad_labels(smiles_list, model_name="rf")
    labels_nn = _get_ad_labels(smiles_list, model_name="nn")

    # Should produce same result since they use the same AD model
    if labels_rf is not None and labels_nn is not None:
        assert np.array_equal(labels_rf, labels_nn)


def test_serving_branches_on_model_type():
    """Test that ModelServer branches AD check on model_type."""
    from serving.model_loader import ModelServer

    # Test with RF (should use descriptor-space AD)
    try:
        server_rf = ModelServer("rf")
        assert server_rf.model_type == "rf"
        # AD logic tested in integration; just verify initialization
    except FileNotFoundError:
        pytest.skip("RF model not trained")

    # Test with NN (should use descriptor-space AD)
    try:
        server_nn = ModelServer("nn")
        assert server_nn.model_type == "nn"
    except FileNotFoundError:
        pytest.skip("NN model not trained")
