import ee
import pandas as pd
import os
import math

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

PROJECT_ID = "wildfire-509104"

SAMPLES_FILE = "data/samples.csv"

BATCH_SIZE = 10000

START_YEAR = 2022
END_YEAR = 2025


# ---------------------------------------------------------
# EARTH ENGINE
# ---------------------------------------------------------

print("Authenticating Earth Engine...")

ee.Authenticate()

ee.Initialize(
    project=PROJECT_ID
)

print("Earth Engine initialized.")


# ---------------------------------------------------------
# LOAD SAMPLES
# ---------------------------------------------------------

print("Loading samples...")

df = pd.read_csv(SAMPLES_FILE)

df["date"] = pd.to_datetime(df["date"])

print("Total samples:", len(df))


# ---------------------------------------------------------
# SENTINEL-2 CLOUD MASK
# ---------------------------------------------------------

def mask_sentinel2(image):

    scl = image.select("SCL")

    # Keep:
    # 4 = vegetation
    # 5 = bare soil
    # 6 = water
    # 7 = unclassified
    #
    # Remove:
    # 3 = cloud shadow
    # 8 = cloud
    # 9 = high probability cloud
    # 10 = cirrus
    # 11 = snow/ice

    mask = (
        scl.eq(4)
        .Or(scl.eq(5))
        .Or(scl.eq(6))
        .Or(scl.eq(7))
    )

    return image.updateMask(mask)


# ---------------------------------------------------------
# ADD NDVI + NDMI
# ---------------------------------------------------------

def add_indices(image):

    ndvi = image.normalizedDifference(
        ["B8", "B4"]
    ).rename("NDVI")

    ndmi = image.normalizedDifference(
        ["B8", "B11"]
    ).rename("NDMI")

    return image.addBands(
        [ndvi, ndmi]
    )


# ---------------------------------------------------------
# PROCESS ONE BATCH
# ---------------------------------------------------------

def process_batch(batch_df, batch_number, year, month):

    print(
        f"Submitting batch {batch_number} "
        f"({len(batch_df)} samples)"
    )

    # -----------------------------------------------------
    # Create Earth Engine FeatureCollection
    # -----------------------------------------------------

    features = []

    for _, row in batch_df.iterrows():

        point = ee.Geometry.Point(
            [
                float(row["longitude"]),
                float(row["latitude"])
            ]
        )

        feature = ee.Feature(
            point,
            {
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "date": row["date"].strftime("%Y-%m-%d"),
                "fire": int(row["fire"])
            }
        )

        features.append(feature)

    points = ee.FeatureCollection(features)

    # -----------------------------------------------------
    # Determine date range
    # -----------------------------------------------------

    dates = batch_df["date"]

    batch_start = dates.min()
    batch_end = dates.max()

    # Use 30 days BEFORE each sample date.
    #
    # Since a batch can contain multiple dates, we need
    # a common Sentinel collection covering the entire
    # possible range.
    collection_start = (
        batch_start - pd.Timedelta(days=30)
    ).strftime("%Y-%m-%d")

    collection_end = (
        batch_end
    ).strftime("%Y-%m-%d")

    # -----------------------------------------------------
    # Sentinel-2 collection
    # -----------------------------------------------------

    collection = (
        ee.ImageCollection(
            "COPERNICUS/S2_SR_HARMONIZED"
        )
        .filterDate(
            collection_start,
            collection_end
        )
        .filterBounds(points)
        .filter(
            ee.Filter.lt(
                "CLOUDY_PIXEL_PERCENTAGE",
                40
            )
        )
        .map(mask_sentinel2)
        .map(add_indices)
    )

    print(
        f"Sentinel window: "
        f"{collection_start} -> {collection_end}"
    )

    # -----------------------------------------------------
    # Median composite
    # -----------------------------------------------------

    composite = collection.select(
        ["NDVI", "NDMI"]
    ).median()

    # -----------------------------------------------------
    # Sample composite at points
    # -----------------------------------------------------

    result = composite.reduceRegions(
        collection=points,
        reducer=ee.Reducer.mean(),
        scale=20
    )

    # -----------------------------------------------------
    # Export
    # -----------------------------------------------------

    description = (
        f"sentinel_{year}_{month:02d}_"
        f"batch_{batch_number:03d}"
    )

    task = ee.batch.Export.table.toDrive(
        collection=result,
        description=description,
        folder="wildfire_sentinel",
        fileNamePrefix=description,
        fileFormat="CSV"
    )

    task.start()

    print(
        "Task started:",
        description
    )


# ---------------------------------------------------------
# PROCESS MONTH BY MONTH
# ---------------------------------------------------------

for year in range(
    START_YEAR,
    END_YEAR + 1
):

    for month in range(1, 13):

        print("\n" + "=" * 70)
        print(
            f"PROCESSING {year}-{month:02d}"
        )
        print("=" * 70)

        month_df = df[
            (df["date"].dt.year == year)
            &
            (df["date"].dt.month == month)
        ].copy()

        if len(month_df) == 0:

            print("No samples.")
            continue

        print(
            "Samples:",
            len(month_df)
        )

        # -------------------------------------------------
        # Split into batches
        # -------------------------------------------------

        num_batches = math.ceil(
            len(month_df) / BATCH_SIZE
        )

        print(
            "Number of batches:",
            num_batches
        )

        for batch_number in range(
            num_batches
        ):

            start = (
                batch_number * BATCH_SIZE
            )

            end = min(
                start + BATCH_SIZE,
                len(month_df)
            )

            batch_df = month_df.iloc[
                start:end
            ]

            process_batch(
                batch_df,
                batch_number + 1,
                year,
                month
            )


print("\nAll Earth Engine tasks submitted.")
print(
    "Open Earth Engine Tasks tab "
    "to monitor them."
)