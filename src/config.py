"""Project-wide settings. Every number here is a modeling choice you should be able to defend."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
PREDICTIONS = ROOT / "predictions"

# The decision horizon. A lease is usually 3-5 years, but most failures happen early,
# and 24 months is long enough to matter and short enough to score within a PhD timeline.
HORIZON_DAYS = 730

# H3 resolution 9 hexes are ~0.1 km^2 (edge ~175 m). A hex plus its ring-1 neighbors
# covers roughly a 5-minute walk, our first approximation of a "walkshed".
H3_RES = 9
WALKSHED_RING = 1
# Resolution 7 parents (~5 km^2, roughly a neighborhood) are the blocks we hold out for spatial
# cross-validation. Res 6 would leave SF with too few blocks for 5 folds.
SPATIAL_BLOCK_RES = 7

# Only businesses that plausibly occupy a storefront. NAICS prefixes:
# 44-45 retail, 722 food & drink, 812 personal services, 7139 fitness/recreation, 4539 misc retail.
STOREFRONT_NAICS_PREFIXES = ("44", "45", "722", "812", "7139")

# Cohorts we learn from. Earlier records are sparse; 2020 is kept but flagged (COVID shock).
STUDY_START_YEAR = 2013
TEMPORAL_TEST_FROM_YEAR = 2021   # train on openings before this, test on openings from this year on

# SF BART stations (approximate centroids, WGS84). Used for transit access until GTFS is wired in.
BART_STATIONS = {
    "Embarcadero": (37.7929, -122.3971),
    "Montgomery": (37.7894, -122.4011),
    "Powell": (37.7844, -122.4079),
    "Civic Center": (37.7795, -122.4138),
    "16th St": (37.7650, -122.4196),
    "24th St": (37.7522, -122.4184),
    "Glen Park": (37.7330, -122.4337),
    "Balboa Park": (37.7215, -122.4474),
}

RANDOM_SEED = 42
