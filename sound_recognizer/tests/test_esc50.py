"""Test for data module."""
from sound_recognizer.data.esc50 import ESC50, ESC50DS
import sound_recognizer.metadata.esc50 as metadata


def test_mapping():
    """Tests that the metadata mapping is correct"""
    ds = ESC50DS(root=metadata.DOWNLOADED_DATA_DIRNAME, download=True)
    assert ds.class_labels == metadata.MAPPING


def test_esc50_data_splits():
    """Fails when any data points are shared between training, test, or validation."""
    dm = ESC50()
    dm.prepare_data()
    dm.setup()
    assert not set(dm.data_train) & set(dm.data_val)
    assert not set(dm.data_train) & set(dm.data_test)
    assert not set(dm.data_val) & set(dm.data_test)
