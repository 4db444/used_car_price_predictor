from pandas import DataFrame
from pathlib import PosixPath

def save_data (df : DataFrame, path : PosixPath, name : str) -> None:
    df.to_csv(path / f"{name}.csv", index=False)