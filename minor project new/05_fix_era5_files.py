import os
import glob
import zipfile
import tempfile

import xarray as xr


ERA5_DIR = "data/era5"


files = sorted(
    glob.glob(
        os.path.join(ERA5_DIR, "era5_*.nc")
    )
)

print("Found", len(files), "ERA5 files")


for zip_file in files:

    filename = os.path.basename(zip_file)

    # Example:
    # era5_2022_01.nc
    name = os.path.splitext(filename)[0]

    output_file = os.path.join(
        ERA5_DIR,
        name + "_fixed.nc"
    )

    if os.path.exists(output_file):
        print("Already exists:", output_file)
        continue

    print("\n" + "=" * 60)
    print("Processing:", filename)
    print("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:

        # ---------------------------------------------
        # Extract ZIP
        # ---------------------------------------------

        with zipfile.ZipFile(zip_file, "r") as z:

            names = z.namelist()

            print("Contents:")
            for n in names:
                print("  ", n)

            z.extractall(temp_dir)

        instant_file = os.path.join(
            temp_dir,
            "data_stream-oper_stepType-instant.nc"
        )

        accum_file = os.path.join(
            temp_dir,
            "data_stream-oper_stepType-accum.nc"
        )

        # ---------------------------------------------
        # Open datasets
        # ---------------------------------------------

        instant = xr.open_dataset(
            instant_file
        )

        accum = xr.open_dataset(
            accum_file
        )

        # ---------------------------------------------
        # Combine
        # ---------------------------------------------

        ds = xr.merge(
            [
                instant,
                accum
            ],
            compat="override"
        )

        print("\nVariables:")
        print(list(ds.data_vars))

        print("\nDimensions:")
        print(ds.dims)

        # ---------------------------------------------
        # Save proper NetCDF
        # ---------------------------------------------

        ds.to_netcdf(
            output_file
        )

        instant.close()
        accum.close()
        ds.close()

    print("Saved:", output_file)


print("\nFinished.")