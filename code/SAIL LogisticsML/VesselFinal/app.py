from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import threading

app = Flask(__name__)


berth_model = None
vessel_model = None
train_model = None

le_port_berth = None
le_mmsi_berth = None
le_mmsi_vessel = None
le_route = None
le_origin_port = None
le_dest_plant = None


port_mmsi_map = {
    "Kolkata": ["419000456", "419001951", "374787000"],
    "Haldia": ["419000456", "419001951", "374787000"],
    "Paradip": ["352978124", "419001155"],
    "Visakhapatnam": ["538004209", "419000183"],
    "Gangavaram": ["419000114", "352122000", "538004587"],
    "Krishnapatnam": ["419000114", "352122000", "538004587"]
}

plants = ["Bokaro", "Durgapur", "Rourkela", "Salem", "VSP"]
train_routes = ["Kolkata → Durgapur", "Visakhapatnam → Bhilai", "Paradip → Rourkela", "Haldia → Burnpur"]


def load_models():
    global berth_model, vessel_model, train_model
    global le_port_berth, le_mmsi_berth, le_mmsi_vessel
    global le_route, le_origin_port, le_dest_plant

    try:
        berth_pack = joblib.load("models/berth_model.pkl")
        vessel_pack = joblib.load("models/vessel_model.pkl")
        train_pack = joblib.load("models/train_model.pkl")

        berth_model = berth_pack["model"]
        le_port_berth = berth_pack["le_port"]
        le_mmsi_berth = berth_pack["le_mmsi"]

        vessel_model = vessel_pack["model"]
        le_mmsi_vessel = vessel_pack["le_mmsi"]
        le_origin_port = vessel_pack["le_origin_port"]
        le_dest_plant = vessel_pack["le_dest_plant"]

        train_model = train_pack["model"]
        le_route = train_pack["le_route"]

        print("✅ All models loaded successfully.")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")

threading.Thread(target=load_models).start()


def safe_transform(le, values):
    transformed = []
    valid_classes = set(le.classes_)
    for v in values:
        if v in valid_classes:
            transformed.append(le.transform([v])[0])
        else:
            transformed.append(le.transform([le.classes_[0]])[0])
    return np.array(transformed)


# ROUTES

@app.route("/")
def index():
    return render_template(
        "index.html",
        ports=list(port_mmsi_map.keys()),
        port_mmsi_map=port_mmsi_map,
        plants=plants,
        train_routes=train_routes
    )


# VESSEL PREDICTION

@app.route("/predict_vessel", methods=["POST"])
def predict_vessel():
    data = request.get_json()
    try:
        origin_port = data["originPort"]
        mmsi = data["mmsi"]
        dest_plant = data["destPlant"]

        df = pd.DataFrame([{
            "mmsi": mmsi,
            "origin_port": origin_port,
            "dest_plant": dest_plant,
            "steel_quantity": 5000,
            "rake_availability": 80,
            "port_utilization": 80,
            "plant_utilization": 75,
            "recent_avg_vessel_delay": 10,
            "distance_km": 450
        }])

        df["mmsi"] = safe_transform(le_mmsi_vessel, df["mmsi"])
        df["origin_port"] = safe_transform(le_origin_port, df["origin_port"])
        df["dest_plant"] = safe_transform(le_dest_plant, df["dest_plant"])

        pred = vessel_model.predict(df)[0]
        return jsonify({"prediction": round(float(pred), 2)})
    except Exception as e:
        return jsonify({"error": str(e)})


# BERTHING PREDICTION

@app.route("/predict_berthing", methods=["POST"])
def predict_berthing():
    data = request.get_json()
    try:
        port = data["port"]
        traffic = float(data["portTraffic"])

        df = pd.DataFrame([{
            "mmsi": "419000456",
            "port": port,
            "port_utilization": traffic,
            "congestion_index": 0.6,
            "recent_avg_delay": 5,
            "arrival_day": 3,
            "vessel_age": 10
        }])
        df["port"] = safe_transform(le_port_berth, df["port"])
        df["mmsi"] = safe_transform(le_mmsi_berth, df["mmsi"])

        pred = berth_model.predict(df)[0]
        return jsonify({"prediction": round(float(pred), 2)})
    except Exception as e:
        return jsonify({"error": str(e)})


# TRAIN PREDICTION

@app.route("/predict_train", methods=["POST"])
def predict_train():
    data = request.get_json()
    try:
        route = data["route"]
        hour = int(data["departureHour"])
        congestion = float(data["networkCongestion"])

        df = pd.DataFrame([{
            "route_encoded": safe_transform(le_route, [route])[0],
            "departure_hour": hour,
            "network_congestion": congestion,
            "plant_utilization": 75,
            "rake_availability": 80,
            "recent_avg_train_delay": 5
        }])

        pred = train_model.predict(df)[0]
        return jsonify({"prediction": round(float(pred), 2)})
    except Exception as e:
        return jsonify({"error": str(e)})


# METRICS ENDPOINT
@app.route("/metrics")
def metrics():
    try:
        berth_df = pd.read_csv("data/berth_delay_data.csv")
        vessel_df = pd.read_csv("data/vessel_delay_data.csv")
        train_df = pd.read_csv("data/train_delay_data.csv")

        avg_vessel = round(vessel_df["vessel_delay_hours"].mean(), 2)
        avg_berth = round(berth_df["berth_delay_hours"].mean(), 2)
        avg_train = round(train_df["train_delay_hours"].mean(), 2)
        busiest_port = berth_df.groupby("port")["berth_delay_hours"].count().idxmax()

        return jsonify({
            "avgVessel": avg_vessel,
            "avgBerth": avg_berth,
            "avgTrain": avg_train,
            "busiestPort": busiest_port
        })
    except Exception as e:
        return jsonify({"error": str(e)})

# MAIN
if __name__ == "__main__":
    app.run(debug=True)
