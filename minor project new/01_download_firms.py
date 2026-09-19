import os
import requests
import pandas as pd
from datetime import datetime, timedelta

from config import (
    FIRMS_MAP_KEY,
    START_DATE,
    END_DATE,
    BBOX
)


# -----------------------------------
# SETTINGS
# -----------------------------------

OUTPUT_DIR = "data/firms"
os.makedirs(OUTPUT_DIR, exist_ok=True)

WEST, SOUTH, EAST, NORTH = BBOX

# We use standard-processing VIIRS S-NPP.
SOURCE = "VIIRS_SNPP_SP"


# -----------------------------------
# Download
# -----------------------------------

start = datetime.strptime(START_DATE, "%Y-%m-%d")
end = datetime.strptime(END_DATE, "%Y-%m-%d")

current = start

all_data = []

while current <= end:

    date_string = current.strftime("%Y-%m-%d")

    print("Downloading:", date_string)

    url = (
        f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
        f"{FIRMS_MAP_KEY}/"
        f"{SOURCE}/"
        f"{WEST},{SOUTH},{EAST},{NORTH}/"
        f"1/{date_string}"
    )

    try:
        response = requests.get(url, timeout=60)

        response.raise_for_status()

        filename = f"{OUTPUT_DIR}/firms_{date_string}.csv"

        with open(filename, "wb") as f:
            f.write(response.content)

        df = pd.read_csv(filename)

        if len(df) > 0:

            all_data.append(df)

            print("  detections:", len(df))

        else:
            print("  no detections")

    except Exception as e:

        print("ERROR:", e)

    current += timedelta(days=1)


# -----------------------------------
# Combine
# -----------------------------------

if all_data:

    final_df = pd.concat(
        all_data,
        ignore_index=True
    )

    final_df.to_csv(
        "data/firms/all_fires.csv",
        index=False
    )

    print()
    print("Total detections:", len(final_df))
    print("Saved to data/firms/all_fires.csv")

else:

    print("No FIRMS data downloaded.")