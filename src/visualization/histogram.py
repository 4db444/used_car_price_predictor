from pandas import DataFrame
from seaborn import histplot
from matplotlib import pyplot
from pathlib import PosixPath

def create_histogram(df : DataFrame, col : str, path : PosixPath) -> None:
    histplot(df, x=col, kde=True)

    pyplot.savefig(path / f"histogram_{col}.png")
    pyplot.close()
