import pandas as pd
import numpy as np

from config import BBOX, NEGATIVE_RATIO


# -----------------------------------
# Load FIRMS
# -----------------------------------

df = pd.read_csv(
    "data/firms/all_fires.csv"
)

print("Original rows:", len(df))


# -----------------------------------
# Keep useful columns
# -----------------------------------

columns = [
    "latitude",
    "longitude",
    "acq_date",
    "acq_time",
    "satellite",
    "confidence",
    "frp",
    "daynight"
]

df = df[columns].copy()


# -----------------------------------
# Convert date
# -----------------------------------

df["date"] = pd.to_datetime(
    df["acq_date"]
)


# -----------------------------------
# Positive label
# -----------------------------------

df["fire"] = 1


# -----------------------------------
# Remove duplicate spatial/date
# detections
# -----------------------------------

df["lat_grid"] = df["latitude"].round(3)
df["lon_grid"] = df["longitude"].round(3)

df = df.drop_duplicates(
    subset=[
        "date",
        "lat_grid",
        "lon_grid"
    ]
)


df = df.drop(
    columns=[
        "lat_grid",
        "lon_grid"
    ]
)


# -----------------------------------
# Save positives
# -----------------------------------

df.to_csv(
    "data/firms/fire_positive.csv",
    index=False
)

print(
    "Positive samples:",
    len(df)
)

# -----------------------------------
# Create negative samples
# -----------------------------------

rng = np.random.default_rng(42)

WEST, SOUTH, EAST, NORTH = BBOX

positive = df.copy()

positive_locations = set(
    zip(
        positive["date"].dt.strftime("%Y-%m-%d"),
        positive["latitude"].round(3),
        positive["longitude"].round(3)
    )
)


negative_rows = []

for _, row in positive.iterrows():

    date = row["date"]

    found = False

    for attempt in range(50):

        # Random nearby location
        lat = row["latitude"] + rng.uniform(-0.15, 0.15)
        lon = row["longitude"] + rng.uniform(-0.15, 0.15)

        # Stay inside region
        if not (
            SOUTH <= lat <= NORTH
            and WEST <= lon <= EAST
        ):
            continue

        key = (
            date.strftime("%Y-%m-%d"),
            round(lat, 3),
            round(lon, 3)
        )

        if key not in positive_locations:

            negative_rows.append({
                "latitude": lat,
                "longitude": lon,
                "date": date,
                "fire": 0
            })

            found = True
            break

    if not found:
        continue


negative = pd.DataFrame(
    negative_rows
)


# -----------------------------------
# Combine
# -----------------------------------

positive["fire"] = 1

dataset = pd.concat(
    [
        positive[
            [
                "latitude",
                "longitude",
                "date",
                "fire"
            ]
        ],
        negative
    ],
    ignore_index=True
)


dataset = dataset.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


dataset.to_csv(
    "data/samples.csv",
    index=False
)


print()
print("Positive:", (dataset["fire"] == 1).sum())
print("Negative:", (dataset["fire"] == 0).sum())
print("Total:", len(dataset))