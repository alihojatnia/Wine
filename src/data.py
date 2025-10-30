import os
import yaml
import click
import pandas as pd
from sklearn.model_selection import train_test_split

def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=";")

def create_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["high_quality"] = (df["quality"] >= 7).astype(int)
    df.drop("quality", axis=1, inplace=True)
    return df

def split_and_save(df: pd.DataFrame, test_size: float, random_state: int):
    train, test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["high_quality"]
    )
    os.makedirs("data/processed", exist_ok=True)
    train.to_csv("data/processed/train.csv", index=False)
    test.to_csv("data/processed/test.csv", index=False)

@click.command()
def preprocess():
    params = yaml.safe_load(open("params.yaml"))["preprocess"]
    raw = load_raw("data/raw/winequality-red.csv")
    df = create_target(raw)
    split_and_save(df, params["test_size"], params["random_state"])

if __name__ == "__main__":
    preprocess()