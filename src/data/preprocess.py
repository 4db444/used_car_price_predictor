from pandas import DataFrame
from src.config import NUMERICAL_COLS, CATEGORICAL_COLS, KEY_GROUPING_COLS

def _calculate_IQR (df : DataFrame, col : str) -> tuple[float, float]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    return Q1 - 1.5 * IQR, Q3 + 1.5 * IQR


def _remove_dup (df : DataFrame) -> DataFrame:
    return df.drop_duplicates()

def _fill_na_numeric_cols(df : DataFrame) -> DataFrame:
    for col in NUMERICAL_COLS:
        df[col] = df.groupby("year")[col].transform(lambda s : s.fillna(s.median()))
        df[col] = df[col].fillna(df[col].median())

    return df

def _fill_na_categorical_cols(df : DataFrame) -> DataFrame:
    for col in CATEGORICAL_COLS:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df

def _fill_na_by_groups(df : DataFrame, key_cols : list[str]) -> DataFrame:
    for col, group_key in KEY_GROUPING_COLS.items():
        df[col] = df.groupby(group_key)[col].ffill()
        df[col] = df.groupby(group_key)[col].bfill()

    return df

def _replace_outliers_with_median(df : DataFrame, col : str) -> DataFrame:
    lower_bound, upper_bound = _calculate_IQR(df, col)

    IQR_median = df.loc[(df[col] >= lower_bound) & (df[col] <= upper_bound), col].median()

    df.loc[(df[col] > upper_bound) | (df[col] < lower_bound), col] = IQR_median

    return df

def _replace_outliers_with_boundry_values(df : DataFrame, col : str) -> DataFrame:
    lower_bound, upper_bound = _calculate_IQR(df, col)

    df.loc[df[col] > upper_bound, col] = upper_bound
    df.loc[df[col] < lower_bound, col] = lower_bound

    return df

def clear_dataframe (df : DataFrame) -> DataFrame: 

    # replace outliers
    df = _replace_outliers_with_median(df, "selling_price")
    df = _replace_outliers_with_median(df, "km_driven")
    df = _replace_outliers_with_boundry_values(df, "year")

    # fill missing values
    df = _fill_na_by_groups(df, KEY_GROUPING_COLS)
    df = _fill_na_categorical_cols(df)
    df = _fill_na_numeric_cols(df)

    # remove duplicates
    df = _remove_dup(df)

    return df