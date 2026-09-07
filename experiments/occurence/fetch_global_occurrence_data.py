"""
Fetch a globally-representative, multi-year dataset for WILDFIRE OCCURRENCE
prediction - v3, with a much stronger, cross-climate-transferable feature
set added on top of the v2 script, in response to poor Leave-One-Region-Out
(LORO) generalization (mean AUC ~0.56) found with the original 9 raw
weather/terrain features.

WHY THE ORIGINAL FEATURES DIDN'T TRANSFER ACROSS REGIONS:
Raw absolute values (temp_mean_k, dewpoint_k) mean different things in
different climates - 305K is an extreme heat anomaly in Siberia but an
ordinary day in central India. A model trained on 9 regions' absolute
values partly learns "what counts as hot/dry HERE", which doesn't
transfer to a 10th region with a different climate baseline.

NEW FEATURES, EACH TARGETING THAT SPECIFIC PROBLEM:

  1. FWI SYSTEM FIRE DANGER INDICES (from CEMS/ECMWF, built on the Canadian
     Fire Weather Index System) - fine_fuel_moisture_code,
     duff_moisture_code, drought_code, initial_fire_spread_index,
     build_up_index, fire_weather_index, keetch_byram_drought_index,
     fire_danger_index. These are PHYSICALLY CALIBRATED to be comparable
     across climates - a Drought Code of 400 means the same thing in
     Siberia and in India. This is the standard global fire-danger system
     used operationally worldwide precisely to solve cross-climate
     comparability, so it's the most direct fix for the LORO problem.
     Source: community GEE catalog, asset
     'projects/climate-engine-pro/assets/ce-cems-fire-daily-4-1'
     (Vitolo et al. 2020, Scientific Data; ECMWF/Copernicus CEMS).

  2. TEMPERATURE ANOMALY (z-score vs. that region's OWN multi-year
     same-month climatology) - encodes "how unusual is this for HERE",
     which is what actually predicts fire risk, rather than an absolute
     temperature that means different things in different climates.

  3. VPD (Vapor Pressure Deficit) - a physically-grounded atmospheric
     dryness measure derived from temperature + dewpoint via the Tetens
     equation. More directly tied to fuel/fire physics than raw
     temperature or dewpoint alone, and its formula is universal (doesn't
     need per-region calibration the way raw temp does).

  4. MULTI-WINDOW PRECIPITATION LAG FEATURES (7d / 30d / 90d sums) and a
     30-day DRY DAY COUNT - genuine time-series/lag features capturing
     drought build-up, instead of a single 14-day snapshot.

LEAKAGE NOTE (since this was asked about explicitly): all lookback windows
use `.filterDate(start, ee.Date(date_str))`, which is a half-open interval
[start, date_str) in Earth Engine - i.e. it EXCLUDES the sample date
itself. The one exception is the FWI system query, which uses
`.filterDate(day.advance(-4,'day'), day.advance(1,'day'))` - this DOES
include the sample date's own daily analysis value, which is intentional
and not leakage: FWI indices are a same-day "nowcast" computed from
weather up to that morning, not a forecast using future information, so
using that day's value is equivalent to knowing "today's fire danger
rating", which is legitimately available before knowing whether a fire
starts.

Setup: same as v2 (earthengine-api, authenticated Google Cloud project).
Everything else (regions, years, per-date export, rate limiting) is
unchanged from v2.
"""

import time
import datetime
import ee

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GEE_PROJECT_ID = "disco-sky-450114-p5"

CURRENT_YEAR = datetime.date.today().year
YEARS_BACK = 6
YEARS = list(range(CURRENT_YEAR - YEARS_BACK, CURRENT_YEAR))

REGIONS = [
    # (name, bbox, season_months, sample_multiplier)
    # sample_multiplier boosts dates/points for regions that came back with
    # too few fires to trust their held-out AUC estimate (iberia: 24 fires,
    # british_columbia: 27, california: 65, southeast_australia: 57 in the
    # last run - all too small for a stable per-region AUC).
    ("central_india",        [78.0, 18.0, 84.5, 24.5],  [2, 3, 4, 5],   1),
    ("california",           [-122.5, 34.0, -117.0, 40.0], [7, 8, 9, 10], 2),
    ("southeast_australia",  [141.0, -38.5, 150.0, -33.0], [11, 12, 1, 2], 2),
    # NOTE: narrowed to exclude Iberia (was [-9.0,...], which almost fully
    # overlapped the iberia bbox below - made that LORO fold non-independent)
    ("mediterranean",        [3.0, 36.0, 24.0, 43.5],   [6, 7, 8, 9],   1),
    ("amazon_arc_of_fire",   [-63.0, -12.0, -46.0, -2.0], [7, 8, 9, 10], 1),
    ("southern_africa",      [20.0, -25.0, 33.0, -15.0], [6, 7, 8, 9],  1),
    ("western_siberia",      [70.0, 55.0, 90.0, 65.0],  [5, 6, 7, 8],   1),
    ("indonesia_sumatra",    [95.0, -6.0, 106.0, 5.0],  [7, 8, 9, 10],  2),
    ("british_columbia",     [-127.0, 49.0, -114.0, 56.0], [6, 7, 8, 9], 2),
    ("iberia",               [-9.5, 37.0, -1.5, 43.5],  [6, 7, 8, 9],   2),
]

DATES_PER_SEASON_PER_YEAR = 2
OFF_SEASON_DATES_PER_YEAR = 1
NUM_SAMPLE_POINTS = 1500
SAMPLE_SCALE_M = 100
LOOKBACK_DAYS = 14
FIRE_LABEL_RADIUS_M = 2000

FWI_ASSET_ID = "projects/climate-engine-pro/assets/ce-cems-fire-daily-4-1"
FWI_BANDS = [
    "fine_fuel_moisture_code", "duff_moisture_code", "drought_code",
    "initial_fire_spread_index", "build_up_index", "fire_weather_index",
    "keetch_byram_drought_index", "fire_danger_index",
]

EXPORT_FOLDER = "wildfire_project_global_v3_1"
TASK_SUBMIT_DELAY_SEC = 1.5


# ---------------------------------------------------------------------------
# Climatology cache (per region, memoized across dates so we don't rebuild
# the same multi-year monthly reduction repeatedly for the same month)
# ---------------------------------------------------------------------------
def get_month_climatology(month, years, cache, band="temperature_2m"):
    key = (band, month)
    if key in cache:
        return cache[key]
    monthly_imgs = []
    for y in years:
        start = ee.Date.fromYMD(y, month, 1)
        end = start.advance(1, "month")
        img = (
            ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
            .filterDate(start, end)
            .select(band)
            .mean()
        )
        monthly_imgs.append(img)
    coll = ee.ImageCollection.fromImages(monthly_imgs)
    clim_mean = coll.mean().rename(f"{band}_clim_mean")
    clim_std = coll.reduce(ee.Reducer.stdDev()).rename(f"{band}_clim_std")
    cache[key] = (clim_mean, clim_std)
    return cache[key]


def compute_vpd(temp_k_img, dewpoint_k_img):
    """Vapor Pressure Deficit via the Tetens equation. Physically grounded,
    universal formula - doesn't need per-region calibration the way raw
    temperature does."""
    temp_c = temp_k_img.subtract(273.15)
    dewpoint_c = dewpoint_k_img.subtract(273.15)
    es = temp_c.expression("0.6108 * exp(17.27 * T / (T + 237.3))", {"T": temp_c})
    ea = dewpoint_c.expression("0.6108 * exp(17.27 * Td / (Td + 237.3))", {"Td": dewpoint_c})
    return es.subtract(ea).rename("vpd_kpa")


def features_for_date(region_geom, date_str, clim_cache):
    end = ee.Date(date_str)
    start = end.advance(-LOOKBACK_DAYS, "day")
    month = int(date_str.split("-")[1])

    era5 = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterDate(start, end)
    temp_mean = era5.select("temperature_2m").mean().rename("temp_mean_k")
    wind_u = era5.select("u_component_of_wind_10m").mean()
    wind_v = era5.select("v_component_of_wind_10m").mean()
    wind_speed = wind_u.hypot(wind_v).rename("wind_speed")
    precip_total = era5.select("total_precipitation_sum").sum().rename("precip_total_m")
    dewpoint = era5.select("dewpoint_temperature_2m").mean().rename("dewpoint_k")

    # --- NEW: max daily wind speed over the lookback window (gust proxy).
    # Short-duration extreme wind events (Santa Ana/Diablo winds in
    # California, hot dry Foehn-type winds in SE Australia) are a primary
    # driver of exactly the fire regime the model struggled with - a
    # 14-day AVERAGE wind speed washes these spikes out entirely. This
    # computes each day's wind speed then takes the max across the window.
    era5_daily_wind = era5.map(
        lambda img: img.select("u_component_of_wind_10m")
            .hypot(img.select("v_component_of_wind_10m"))
            .rename("daily_wind_speed")
    )
    wind_speed_max = era5_daily_wind.max().rename("wind_speed_max")

    # --- NEW: wind anomaly vs this region's own multi-year climatology,
    # same idea as temp_anomaly_z - "how unusual is peak wind FOR HERE" ---
    wind_clim_mean, wind_clim_std = get_month_climatology(
        month, YEARS, clim_cache, band="u_component_of_wind_10m"
    )
    # crude but workable: compare max wind speed to the mean U-wind climatology
    # magnitude (sign doesn't matter here, we care about wind intensity anomaly)
    wind_anomaly_z = wind_speed_max.subtract(wind_clim_mean.abs()).divide(
        wind_clim_std.add(0.1)
    ).rename("wind_anomaly_z")

    # --- NEW: temperature anomaly vs this region's own multi-year climatology ---
    clim_mean, clim_std = get_month_climatology(month, YEARS, clim_cache)
    temp_anomaly_z = temp_mean.subtract(clim_mean).divide(clim_std.add(0.1)).rename("temp_anomaly_z")

    # --- NEW: VPD ---
    vpd = compute_vpd(temp_mean, dewpoint)

    # --- NEW: multi-window precipitation lag features ---
    precip_7d = (
        ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .filterDate(end.advance(-7, "day"), end)
        .select("total_precipitation_sum").sum().rename("precip_sum_7d")
    )
    precip_30d_coll = (
        ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .filterDate(end.advance(-30, "day"), end)
    )
    precip_30d = precip_30d_coll.select("total_precipitation_sum").sum().rename("precip_sum_30d")
    dry_day_count_30d = (
        precip_30d_coll.select("total_precipitation_sum")
        .map(lambda img: img.lt(0.001))
        .sum().rename("dry_day_count_30d")
    )
    precip_90d = (
        ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .filterDate(end.advance(-90, "day"), end)
        .select("total_precipitation_sum").sum().rename("precip_sum_90d")
    )
    temp_mean_30d = (
        ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .filterDate(end.advance(-30, "day"), end)
        .select("temperature_2m").mean().rename("temp_mean_30d")
    )

    # --- NEW: FWI System fire danger indices (physically calibrated,
    # comparable across climates - the main fix for LORO generalization) ---
    fwi_coll = (
        ee.ImageCollection(FWI_ASSET_ID)
        .filterDate(end.advance(-4, "day"), end.advance(1, "day"))
        .sort("system:time_start", False)
    )
    fwi_img = fwi_coll.first().select(FWI_BANDS)

    # --- Vegetation / fuel proxy ---
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region_geom)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
    )
    ndvi = s2.median().normalizedDifference(["B8", "B4"]).rename("ndvi")

    landcover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").rename("landcover")

    dem = ee.Image("USGS/SRTMGL1_003")
    slope = ee.Terrain.slope(dem).rename("slope_deg")
    aspect = ee.Terrain.aspect(dem).rename("aspect_deg")
    elevation = dem.rename("elevation_m")

    return (
        temp_mean.addBands(wind_speed).addBands(precip_total).addBands(dewpoint)
        .addBands(wind_speed_max).addBands(wind_anomaly_z)
        .addBands(temp_anomaly_z).addBands(vpd)
        .addBands(precip_7d).addBands(precip_30d).addBands(precip_90d)
        .addBands(dry_day_count_30d).addBands(temp_mean_30d)
        .addBands(fwi_img)
        .addBands(ndvi).addBands(landcover).addBands(slope).addBands(aspect).addBands(elevation)
    )


def fire_label_image(date_str):
    day = ee.Date(date_str)
    firms = ee.ImageCollection("FIRMS").filterDate(day.advance(-1, "day"), day.advance(1, "day"))
    fire_present = firms.select("T21").mosaic().gt(0).unmask(0)
    return fire_present.focal_max(radius=FIRE_LABEL_RADIUS_M, units="meters").rename("fire_label")


def submit_task(region_geom, region_name, date_str, is_off_season, clim_cache, num_points):
    feats = features_for_date(region_geom, date_str, clim_cache)
    label = fire_label_image(date_str)
    combined = feats.addBands(label)

    sampled = combined.sample(
        region=region_geom,
        scale=SAMPLE_SCALE_M,
        numPixels=num_points,
        seed=42,
        geometries=True,
        tileScale=4,
    )
    sampled = sampled.map(lambda f, d=date_str, r=region_name: f.set("date", d).set("region", r))

    safe_date = date_str.replace("-", "")
    tag = "offseason" if is_off_season else "season"
    description = f"wfv3_{region_name}_{safe_date}_{tag}"

    task = ee.batch.Export.table.toDrive(
        collection=sampled,
        description=description,
        folder=EXPORT_FOLDER,
        fileFormat="CSV",
    )
    task.start()
    print(f"Submitted: {description}")


def season_sample_dates(months, year, n):
    chosen_months = months[:: max(1, len(months) // n)][:n]
    dates = []
    for m in chosen_months:
        y = year if m >= months[0] or months[0] <= months[-1] else year + 1
        dates.append(f"{y}-{m:02d}-10")
    return dates


def off_season_months(season_months):
    all_months = set(range(1, 13))
    return sorted(all_months - set(season_months))


def main():
    ee.Initialize(project=GEE_PROJECT_ID)

    total_tasks = 0
    for region_name, bbox, season_months, sample_multiplier in REGIONS:
        region_geom = ee.Geometry.Rectangle(bbox)
        clim_cache = {}
        num_points = NUM_SAMPLE_POINTS * sample_multiplier
        dates_per_season = DATES_PER_SEASON_PER_YEAR * sample_multiplier

        for year in YEARS:
            for date_str in season_sample_dates(season_months, year, dates_per_season):
                submit_task(region_geom, region_name, date_str, is_off_season=False,
                            clim_cache=clim_cache, num_points=num_points)
                total_tasks += 1
                time.sleep(TASK_SUBMIT_DELAY_SEC)

            off_months = off_season_months(season_months)
            for date_str in season_sample_dates(off_months, year, OFF_SEASON_DATES_PER_YEAR):
                submit_task(region_geom, region_name, date_str, is_off_season=True,
                            clim_cache=clim_cache, num_points=num_points)
                total_tasks += 1
                time.sleep(TASK_SUBMIT_DELAY_SEC)

    print(f"\nSubmitted {total_tasks} export tasks total across {len(REGIONS)} regions "
          f"and {len(YEARS)} years.")
    print("Monitor progress at: https://code.earthengine.google.com/tasks")
    print(f"Download all completed CSVs from the '{EXPORT_FOLDER}' Drive folder into one "
          f"local folder, then run combine_downloaded_csvs('<that folder>').")


def combine_downloaded_csvs(folder_path, output_path="./wildfire_occurrence_global_v3_1.csv"):
    import glob
    import pandas as pd
    csv_files = glob.glob(f"{folder_path}/*.csv")
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {folder_path}")
    dfs = [pd.read_csv(f) for f in csv_files]
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_csv(output_path, index=False)
    print(f"Combined {len(csv_files)} files ({len(combined_df)} rows total) into {output_path}")


if __name__ == "__main__":
    main()