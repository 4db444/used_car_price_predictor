from pandas import DataFrame, Series
from pathlib import PosixPath
from ..config import CATEGORICAL_COLS

def _descriptive_analysis (df : DataFrame) -> DataFrame:
    return df.describe()

def _category_frequency (df : DataFrame, columns : list[str]) -> dict[str, Series]:
    return {col : df[col].value_counts() for col in columns}

def _na_frequency (df : DataFrame) -> DataFrame:
    length = len(df)
    miss = df.isna().sum()
    frq = (miss / length) * 100

    return DataFrame({"missing" : miss, "missing_percentage" : frq})

def _corr_matrix (df : DataFrame): 
    return df.corr(numeric_only=True)

def full_analyse (df : DataFrame, path : PosixPath) -> dict[str, DataFrame]:

    analyse_result = {
        "descriptive_analysis" : _descriptive_analysis(df),
        "na_frequency" : _na_frequency(df),
        "corr_matrix" : _corr_matrix(df)
    }

    for key, value in analyse_result.items():
        value.to_csv(path / f"{key}.csv")
    
    return analyse_result