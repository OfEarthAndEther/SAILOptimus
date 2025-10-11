import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib, os

os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/train_delay_data.csv")
le_route = LabelEncoder()
df["route_encoded"] = le_route.fit_transform(df["route"])

X = df[["route_encoded", "departure_hour", "network_congestion",
        "plant_utilization", "rake_availability", "recent_avg_train_delay"]]
y = df["train_delay_hours"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

joblib.dump({"model": model, "le_route": le_route}, "models/train_model.pkl")
print("✅ train_model saved.")
