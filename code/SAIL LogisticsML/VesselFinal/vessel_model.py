import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib, os

os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/vessel_delay_data.csv")
le_port = LabelEncoder()
le_plant = LabelEncoder()
le_mmsi = LabelEncoder()

df["origin_port"] = le_port.fit_transform(df["origin_port"])
df["dest_plant"] = le_plant.fit_transform(df["dest_plant"])
df["mmsi"] = le_mmsi.fit_transform(df["mmsi"])

X = df[[
    "mmsi", "origin_port", "dest_plant", "steel_quantity",
    "rake_availability", "port_utilization", "plant_utilization",
    "recent_avg_vessel_delay", "distance_km"
]]
y = df["vessel_delay_hours"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

joblib.dump({
    "model": model,
    "le_origin_port": le_port,
    "le_dest_plant": le_plant,
    "le_mmsi": le_mmsi
}, "models/vessel_model.pkl")

print("✅ vessel_model saved.")
