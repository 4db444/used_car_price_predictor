from datetime import datetime
from pandas import DataFrame, get_dummies
from streamlit import (
    set_page_config,
    title,
    markdown,
    sidebar,
    subheader,
    metric,
    error,
)
from joblib import load

from src.config import FINAL_MODEL_PATH


set_page_config(page_title="Used Car Price Predictor", layout="centered")

title("Used Car Price Predictor")
markdown(
    "Enter the vehicle characteristics on the left to get an instant estimated "
    "selling price."
)

with open(FINAL_MODEL_PATH, "rb") as model_file:
    artifacts = load(model_file)

model = artifacts["model"]
feature_columns = artifacts["feature_columns"]

BRANDS = [
    "Ambassador", "Audi", "BMW", "Chevrolet", "Daewoo", "Datsun", "Fiat",
    "Force", "Ford", "Honda", "Hyundai", "Isuzu", "Jaguar", "Jeep", "Kia",
    "Land", "MG", "Mahindra", "Maruti", "Mercedes-Benz", "Mitsubishi",
    "Nissan", "OpelCorsa", "Renault", "Skoda", "Tata", "Toyota",
    "Volkswagen", "Volvo",
]

sidebar.markdown("### Vehicle features")

year = sidebar.number_input(
    "Year", min_value=1990, max_value=datetime.now().year, value=2015
)
km_driven = sidebar.number_input("Kilometers driven", min_value=0, value=50000)
brand = sidebar.selectbox("Brand", BRANDS)

fuel = sidebar.selectbox("Fuel", ["CNG", "Diesel", "Electric", "LPG", "Petrol"])
transmission = sidebar.selectbox("Transmission", ["Automatic", "Manual"])
owner = sidebar.selectbox(
    "Owner",
    [
        "First Owner",
        "Second Owner",
        "Third Owner",
        "Fourth & Above Owner",
        "Test Drive Car",
    ],
)
seller_type = sidebar.selectbox(
    "Seller type", ["Dealer", "Individual", "Trustmark Dealer"]
)

if sidebar.button("Estimate price"):
    try:
        row = DataFrame([{
            "name": brand,
            "year": float(year),
            "km_driven": float(km_driven),
            "fuel": fuel,
            "owner": owner,
            "transmission": transmission,
            "seller_type": seller_type,
            "selling_price": 0,
        }])

        row["age"] = datetime.now().year - row["year"]
        row["brand"] = row["name"]

        row = get_dummies(
            row,
            columns=["fuel", "owner", "transmission", "brand", "seller_type"],
        )
        row = row.reindex(columns=feature_columns, fill_value=0)

        prediction = model.predict(row)[0]

        subheader("Estimated price")
        metric(label=f"Predicted selling price ({brand}, {int(km_driven)} km)",
               value=f"{prediction:,.0f} $")
    except Exception as exc:
        error(f"An error occurred: {exc}")