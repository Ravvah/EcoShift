from datetime import datetime, timedelta, timezone
import httpx
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_URL = "http://127.0.0.1:8000/api/v1/predict"
API_KEY = ""

now = datetime.now(timezone.utc)
history = []

for i in range(384):
    ts = now - timedelta(minutes=30 * (384 - i))
    history.append({
        "timestamp": ts.isoformat(),
        "price_eur_mwh": 50.0 +(i % 20),
        "co2_intensity_g_kwh": 30.0 + (i% 15)
    })

payload = {
    "horizon_hours": 24,
    "history": history
}

headers = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

logger.info(f"Testing inference endpoint with {len(history)} points to {API_URL}")

try:
    with httpx.Client(timeout=10.0) as client:
        response = client.post(url=API_URL, json=payload, headers=headers)

    logger.info(f"HTTP Status : {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        logger.info(f"Inference succeed \n-Requested horizon : {payload['horizon_hours']}h \n- Number of price predictions : {len(data.get('predictions', []))} \n- First prediction example : {data['predictions'][0]}")

    else:
        logger.info(f"Error : {response.status_code} : {response.text}")

except httpx.ConnectError:
    logger.info("\nCloud not connect to server")