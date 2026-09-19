import os
import cdsapi
import calendar

from config import (
    START_DATE,
    END_DATE,
    ERA5_AREA
)


# ==========================================
# SETTINGS
# ==========================================

OUTPUT_DIR = "data/era5"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ==========================================
# CDS CLIENT
# ==========================================

client = cdsapi.Client()


# ==========================================
# YEARS
# ==========================================

start_year = int(START_DATE[:4])
end_year = int(END_DATE[:4])


# ==========================================
# MONTHS
# ==========================================

months = [
    f"{i:02d}"
    for i in range(1, 13)
]


# ==========================================
# HOURS
# ==========================================

hours = [
    f"{i:02d}:00"
    for i in range(24)
]


# ==========================================
# VARIABLES
# ==========================================

variables = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_precipitation"
]


# ==========================================
# DOWNLOAD
# ==========================================

for year in range(start_year, end_year + 1):
    for month in months:

        # Number of days in this month
        num_days = calendar.monthrange(
            year,
            int(month)
        )[1]

        days = [
            f"{day:02d}"
            for day in range(1, num_days + 1)
        ]

        output = os.path.join(
            OUTPUT_DIR,
            f"era5_{year}_{month}.nc"
        )

        # ----------------------------------
        # Skip if already downloaded
        # ----------------------------------

        if os.path.exists(output):

            print(
                f"Already exists: {output}"
            )

            continue


        print()
        print("=" * 50)
        print(
            f"Downloading ERA5: "
            f"{year}-{month}"
        )
        print("=" * 50)


        try:

            client.retrieve(

                "reanalysis-era5-single-levels",

                {

                    "product_type": "reanalysis",

                    "variable": variables,

                    "year": str(year),

                    "month": month,

                    "day": days,

                    "time": hours,

                    "area": ERA5_AREA,

                    "format": "netcdf"

                },

                output
            )


            print(
                f"Saved: {output}"
            )


        except Exception as e:

            print()
            print(
                f"FAILED: {year}-{month}"
            )

            print(e)

            print(
                "Moving to next month..."
            )

            continue