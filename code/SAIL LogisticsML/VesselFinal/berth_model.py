import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib, os

os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/berth_delay_data.csv")
le_port = LabelEncoder()
le_mmsi = LabelEncoder()
df["port"] = le_port.fit_transform(df["port"])
df["mmsi"] = le_mmsi.fit_transform(df["mmsi"])

X = df[["mmsi", "port", "port_utilization", "congestion_index",
        "recent_avg_delay", "arrival_day", "vessel_age"]]
y = df["berth_delay_hours"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

joblib.dump({"model": model, "le_port": le_port, "le_mmsi": le_mmsi}, "models/berth_model.pkl")
print("✅ berth_model saved.")
