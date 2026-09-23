from pandas import DataFrame
from seaborn import scatterplot
from matplotlib import pyplot
from pathlib import PosixPath

def create_scatter_plot(df : DataFrame, x_col : str, y_col : str, path : PosixPath) -> None:
    scatterplot(df, x=x_col, y=y_col)

    pyplot.savefig(path / f"scatter_plot_{x_col}_{y_col}.png")
    pyplot.close()
