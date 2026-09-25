from pandas import DataFrame
from datetime import datetime

def _create_age_col (df : DataFrame) -> DataFrame:

    if "age" not in df.columns:
        current_year = datetime.now().year
        df["age"] = current_year - df["year"]

    return df

def _create_brand_col (df : DataFrame) -> DataFrame:

    if "brand" not in df.columns:
        brands_serie = df["name"].apply(lambda elem : elem.split(" ")[0])
        df["brand"] = brands_serie

    return df

def create_features (df : DataFrame) -> DataFrame:

    df = _create_age_col(df)
    df = _create_brand_col(df)

    return df