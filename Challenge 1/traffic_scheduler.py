import schedule
import time
import requests
import pandas as pd
import os
from datetime import datetime as dt

# --- CONFIGURATION ---
TOMTOM_API_KEY = "AADsaSybeHMMub2H3TzHoO4AvuK8XKnX"
base_url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
save_dir = r"C:\AI\Individual\Challenge 1"
os.makedirs(save_dir, exist_ok=True)

locations = {
    "CityCenter": (42.6977, 23.3219),
    "Tsarigradsko": (42.6589, 23.3822),
    "Lyulin": (42.7160, 23.2546),
    "StudentskiGrad": (42.6500, 23.3500)
}

# --- FUNCTION TO FETCH TRAFFIC ---
def fetch_traffic():
    timestamp = dt.now().replace(minute=0, second=0, microsecond=0)
    print(f"\n🚦 Collecting traffic data at {timestamp}")

    records = []
    for name, (lat, lon) in locations.items():
        try:
            url = f"{base_url}?point={lat},{lon}&key={TOMTOM_API_KEY}"
            r = requests.get(url, timeout=10)
            data_json = r.json()

            if "flowSegmentData" in data_json:
                d = data_json["flowSegmentData"]
                record = {
                    "time": timestamp,
                    "location": name,
                    "current_speed": d.get("currentSpeed"),
                    "free_flow_speed": d.get("freeFlowSpeed"),
                    "confidence": d.get("confidence"),
                    "road_closure": d.get("roadClosure", False)
                }
                records.append(record)
                print(f"✅ {name}: {d.get('currentSpeed')} km/h (confidence {d.get('confidence')})")
            else:
                print(f"⚠️ No data for {name}")
        except Exception as e:
            print(f"❌ Error for {name}: {e}")

    if records:
        df = pd.DataFrame(records)
        df["traffic_slowdown"] = df["free_flow_speed"] - df["current_speed"]
        df["traffic_pressure"] = df["traffic_slowdown"] * df["confidence"]

        filename = os.path.join(save_dir, "sofia_hourly_traffic.csv")
        df.to_csv(filename, mode="a", header=not os.path.exists(filename), index=False)
        print(f"💾 Logged {len(df)} records → {filename}")
    else:
        print("⚠️ No records logged this hour")


# --- SCHEDULE EVERY HOUR ---
schedule.every(1).hours.do(fetch_traffic)

print("🚗 Traffic Scheduler started — will run every hour (Ctrl + C to stop)\n")

# --- CONTINUOUS LOOP ---
fetch_traffic()   # run immediately once at start
while True:
    schedule.run_pending()
    time.sleep(10)
