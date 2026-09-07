"""
Fetch PRE-FIRE and POST-FIRE Sentinel-2 image pairs for known fire events,
and derive burn SEVERITY labels programmatically using dNBR (the
difference Normalized Burn Ratio) - the standard remote-sensing technique
for this, so you don't need to hand-label anything.

NBR  = (NIR - SWIR2) / (NIR + SWIR2)     [uses Sentinel-2 bands B8, B12]
dNBR = NBR_pre - NBR_post

Severity classes (standard USGS/UN-SPIDER thresholds, approximate):
    0: unburned         dNBR < 0.10
    1: low severity      0.10 <= dNBR < 0.27
    2: moderate-low       0.27 <= dNBR < 0.44
    3: moderate-high      0.44 <= dNBR < 0.66
    4: high severity     dNBR >= 0.66

Output per fire event: a stacked GeoTIFF with
    - 6 pre-fire bands (matching your existing model's input format)
    - 6 post-fire bands
    - 1 severity label band (0-4)
exported to your Google Drive.

Setup: same as fetch_occurrence_data.py (earthengine-api + authenticated
Google Cloud project).
"""

import ee

# ---------------------------------------------------------------------------
# Config - EDIT THESE
# ---------------------------------------------------------------------------
GEE_PROJECT_ID = "your-gcp-project-id"

# List of known fire events: (name, [min_lon,min_lat,max_lon,max_lat], pre_date, post_date)
# pre_date/post_date should bracket the fire - e.g. pre = a month before fire
# season, post = shortly after containment. Fill in real events for your
# region (news reports, FSI forest fire alerts, or FIRMS hotspot clusters
# can help you identify dates/locations).
FIRE_EVENTS = [
    (
        "example_event_1",
        [81.5, 20.5, 81.8, 20.8],   # bbox around the burn area
        "2025-02-01",                # pre-fire date (before ignition)
        "2025-05-01",                # post-fire date (after burn, before regrowth)
    ),
    # Add more events here - more events = more training data
]

# Sentinel-2 bands to keep, matching your original 6-band setup
S2_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]  # Blue, Green, Red, NIR, SWIR1, SWIR2
MAX_CLOUD_PCT = 20
EXPORT_SCALE_M = 10
EXPORT_FOLDER = "wildfire_severity_project"


def get_clearest_image(bbox_geom, date_str, window_days=30):
    """Grab the least-cloudy Sentinel-2 image within +/- window_days of date_str."""
    center = ee.Date(date_str)
    coll = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(bbox_geom)
        .filterDate(center.advance(-window_days, "day"), center.advance(window_days, "day"))
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD_PCT))
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )
    return coll.first().select(S2_BANDS)


def compute_nbr(img):
    return img.normalizedDifference(["B8", "B12"]).rename("nbr")


def severity_from_dnbr(dnbr):
    """Classify dNBR into 5 severity classes using standard thresholds."""
    severity = (
        ee.Image(0)
        .where(dnbr.gte(0.10).And(dnbr.lt(0.27)), 1)
        .where(dnbr.gte(0.27).And(dnbr.lt(0.44)), 2)
        .where(dnbr.gte(0.44).And(dnbr.lt(0.66)), 3)
        .where(dnbr.gte(0.66), 4)
    )
    return severity.rename("severity").toByte()


def process_event(name, bbox, pre_date, post_date):
    region = ee.Geometry.Rectangle(bbox)

    pre_img = get_clearest_image(region, pre_date)
    post_img = get_clearest_image(region, post_date)

    pre_nbr = compute_nbr(pre_img)
    post_nbr = compute_nbr(post_img)
    dnbr = pre_nbr.subtract(post_nbr)
    severity = severity_from_dnbr(dnbr)

    pre_renamed = pre_img.rename([f"pre_{b}" for b in S2_BANDS])
    post_renamed = post_img.rename([f"post_{b}" for b in S2_BANDS])

    stacked = pre_renamed.addBands(post_renamed).addBands(severity)

    task = ee.batch.Export.image.toDrive(
        image=stacked,
        description=f"wildfire_severity_{name}",
        folder=EXPORT_FOLDER,
        region=region,
        scale=EXPORT_SCALE_M,
        maxPixels=1e10,
        fileFormat="GeoTIFF",
    )
    task.start()
    print(f"Submitted export task for '{name}'.")


def main():
    ee.Initialize(project=GEE_PROJECT_ID)
    for name, bbox, pre_date, post_date in FIRE_EVENTS:
        process_event(name, bbox, pre_date, post_date)

    print("\nAll export tasks submitted.")
    print("Monitor progress at: https://code.earthengine.google.com/tasks")
    print(f"Download completed GeoTIFFs from the '{EXPORT_FOLDER}' folder in "
          f"your Google Drive, then place them in a local folder for training.")


if __name__ == "__main__":
    main()
