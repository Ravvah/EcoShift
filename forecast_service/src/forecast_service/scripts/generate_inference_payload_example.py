from datetime import datetime, timedelta, timezone
import json
import math

start_time = datetime.now(timezone.utc) - timedelta(hours=672)
history = []

for i in range(672):
    dt = start_time + timedelta(hours=i)
    hour = dt.hour

    # Synthetic data (peak at 18:00, trough at 03:00)
    base_price = 70.0 + 35.0 * math.sin((hour - 6) * math.pi / 12)
    base_co2 = 130.0 + 40.0 * math.sin((hour - 6) * math.pi / 12)

    history.append(
        {
            "timestamp": dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "price_eur_mwh": round(max(10.0, base_price), 2),
            "co2_intensity_g_kwh": round(max(50.0, base_co2), 1),
        }
    )

payload = {"horizon_hours": 24, "history": history}

with open("payload_672.json", "w") as f:
    json.dump(payload, f, indent=2)

print(f"Generated payload with {len(history)} entries in 'payload_672.json'.")