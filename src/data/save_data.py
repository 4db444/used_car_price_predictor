from pandas import DataFrame
from pathlib import PosixPath

def save_data (df : DataFrame, path : PosixPath, name : str, index : bool = False) -> None:
    df.to_csv(path / f"{name}.csv", index=index)