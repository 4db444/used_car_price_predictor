from pandas import DataFrame
from pathlib import PosixPath

def save_data (df : DataFrame, path : PosixPath) -> None:
    df.to_csv(path / "data_cleaned.csv", index=False)