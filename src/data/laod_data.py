from pandas import DataFrame, read_csv
from pathlib import PosixPath

def load_data(path : PosixPath) -> DataFrame:
    return read_csv(path)