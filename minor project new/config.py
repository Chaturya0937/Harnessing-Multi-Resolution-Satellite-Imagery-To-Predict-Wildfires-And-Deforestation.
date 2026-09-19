# =========================
# PROJECT CONFIGURATION
# =========================

START_DATE = "2022-01-01"
END_DATE = "2025-12-31"

# Chhattisgarh approximate bounding box
# west, south, east, north
BBOX = (80.2, 17.8, 84.3, 24.1)

# NASA FIRMS
FIRMS_MAP_KEY = "a009baeef7a516f21904c6bd5c3fef9e"

# ERA5
ERA5_AREA = [
    24.1,   # north
    80.2,   # west
    17.8,   # south
    84.3    # east
]

# Number of negative samples per positive sample
NEGATIVE_RATIO = 1

# Sentinel-2 cloud threshold
MAX_CLOUD = 40