from seaborn import heatmap
from pathlib import PosixPath
from pandas import DataFrame
from matplotlib import pyplot

def create_heatmap(df : DataFrame, path : PosixPath, name : str = "") -> None: 
    heatmap(
        df, 
        annot=True,
        cmap="coolwarm"
    )

    pyplot.savefig(path / f"heatmap_{name}.png")

    pyplot.close()
