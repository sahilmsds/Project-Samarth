import requests
import pandas as pd
import time
from dotenv import load_dotenv
import os
load_dotenv
# --- CONFIG ---
API_KEY = os.getenv("API_KEY")
RESOURCE_ID = os.getenv("RESOURCE_ID")
BASE_URL = os.getenv("BASEURL")
LIMIT = 10000  # Max allowed per request
OUTPUT_FILE = "crop_production_data.csv"

print("Starting data download from data.gov.in...\n")

all_records = []
offset = 0
total_records = None

while True:
    # API request
    params = {
        "api-key": API_KEY,
        "format": "json",
        "limit": LIMIT,
        "offset": offset,
    }

    try:
        res = requests.get(BASE_URL, params=params, timeout=60)
        res.raise_for_status()
        data = res.json()
    except Exception as e:
        print(f"Error fetching data at offset {offset}: {e}")
        time.sleep(5)
        continue

    records = data.get("records", [])
    if not records:
        print("No more records to fetch.")
        break

    all_records.extend(records)
    total_records = data.get("total", len(all_records))
    print(f"Fetched {len(records)} records (total so far: {len(all_records)} / {total_records})")

    offset += LIMIT
    time.sleep(1)  # small delay to be API-friendly

    # stop when fetched all
    if len(all_records) >= total_records:
        break

print("\nDownload complete! Total records fetched:", len(all_records))

# --- Save Raw Data ---
df = pd.DataFrame(all_records)
df.to_csv("raw_crop_production_data.csv", index=False)
print("Saved raw data to 'raw_crop_production_data.csv'")

# --- Normalize Columns ---
rename_map = {
    "State_Name": "state",
    "District_Name": "district",
    "Crop_Year": "year",
    "Season": "season",
    "Crop": "crop",
    "Area": "area",
    "Production": "production",
}
df = df.rename(columns=rename_map)

# Only keep relevant columns
cols_to_keep = ["state", "district", "year", "season", "crop", "area", "production"]
df = df[[c for c in cols_to_keep if c in df.columns]]

# Clean data
if "state" in df.columns:
    df["state"] = df["state"].str.title().str.strip()
if "district" in df.columns:
    df["district"] = df["district"].str.title().str.strip()
if "season" in df.columns:
    df["season"] = df["season"].str.title().str.strip()
if "crop" in df.columns:
    df["crop"] = df["crop"].str.title().str.strip()

df.to_csv(OUTPUT_FILE, index=False)
print(f"Cleaned data saved to '{OUTPUT_FILE}'")
print("All done! You can now start your analysis.")
