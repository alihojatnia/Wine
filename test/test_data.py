import pandas as pd
from src.data import load_raw, create_target

def test_load_raw():
    df = load_raw("data/raw/winequality-red.csv")
    assert df.shape == (1599, 12)

def test_target():
    df = load_raw("data/raw/winequality-red.csv")
    df = create_target(df)
    assert "high_quality" in df.columns
    assert df["high_quality"].isin([0, 1]).all()