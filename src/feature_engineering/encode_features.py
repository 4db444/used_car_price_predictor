from pandas import DataFrame, get_dummies

def encode_features (df : DataFrame) -> DataFrame:

    df = get_dummies(
        df,
        columns=["fuel", "owner", "transmission", "brand", "seller_type"],
        dtype=int
    )
    
    return df