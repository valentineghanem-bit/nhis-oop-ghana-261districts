"""
01_data_wrangling.py
=====================
NHIS OOP Ghana 261 Districts -- Phase 2 Data Wrangling
Author: Valentine Golden Ghanem | AIPOCH v6.5 | 2026-05-28

Tasks:
  1. Load district_master_261.csv (261 x 24)
  2. Build region-to-district crosswalk
  3. Merge DHS 2019 NHIS regional coverage (validation)
  4. Merge DHS 2022 access barriers (regional -> district)
  5. Merge DHS 2022 socioeconomic indicators (regional -> district)
  6. Compute derived binary/categorical variables
  7. Build spatial weights matrix (Queen contiguity, 260 districts with geometry)
  8. Generate Table 1 (descriptive statistics by ecological zone, N=261)
  9. Save district_master_261_analytical.csv (final pre-modelling dataset)
 10. Save spatial_weights_queen_260.pkl
"""

import os, json, pickle, warnings
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW  = os.path.join(BASE, 'data', 'raw')
PROC = os.path.join(BASE, 'data', 'processed')

audit = []
def log(msg):
    print(msg)
    audit.append(msg)

log("=" * 70)
log("DATA WRANGLING -- NHIS OOP Ghana 261 Districts")
log("AIPOCH v6.5 | 2026-05-28")
log("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 -- Load primary district dataset
# ─────────────────────────────────────────────────────────────────────────────
log("\n[STEP 1] Loading district_master_261.csv")
dm = pd.read_csv(os.path.join(PROC, 'district_master_261.csv'))
log(f"  Shape: {dm.shape}")
log(f"  Has_Geometry=True:  {dm['Has_Geometry'].sum()}")
log(f"  Has_Geometry=False: {(~dm['Has_Geometry']).sum()}")
assert dm.shape == (261, 24), f"Unexpected shape: {dm.shape}"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 -- Build region-to-GeoJSON-region crosswalk
# The GeoJSON uses 'GEO_REGION'; DHS uses region names. Build a normalised
# mapping so DHS regional variables can be assigned to districts.
# ─────────────────────────────────────────────────────────────────────────────
log("\n[STEP 2] Building region crosswalk (GEO_REGION <-> DHS Region_DHS)")

# Inspect GEO_REGION values
geo_regions = dm['GEO_REGION'].unique()
log(f"  GEO_REGION values ({len(geo_regions)}): {sorted(str(r) for r in geo_regions)}")

# Load DHS files to check Region_DHS values
nhis = pd.read_csv(os.path.join(PROC, 'nhis_regional_2019.csv'))
acc  = pd.read_csv(os.path.join(PROC, 'access_regional_2022.csv'))
soc  = pd.read_csv(os.path.join(PROC, 'socioeconomic_regional_2022.csv'))

dhs_regions_nhis = sorted(nhis['Region_DHS'].unique())
dhs_regions_acc  = sorted(acc['Region_DHS'].unique())
log(f"  DHS NHIS regions ({len(dhs_regions_nhis)}): {dhs_regions_nhis}")
log(f"  DHS Access regions ({len(dhs_regions_acc)}): {dhs_regions_acc}")

# GeoJSON uses 2019 regional names (10 regions pre-2019 split / 16 post).
# DHS 2019 uses post-2019 names (up to 16 regions after splits).
# Build mapping: GEO_REGION -> DHS Region_DHS (for NHIS 2019 = 11 regions)
# and GEO_REGION -> DHS Region_DHS (for Access 2022 = 21 regions)

# The GeoJSON regions reflect the 2022 boundary file — check actual values
# and build normalised mapping.
def norm(s):
    return str(s).strip().upper()

# Build GEO -> NHIS lookup: normalise both sides
nhis_lookup = {norm(r): r for r in nhis['Region_DHS']}
acc_lookup  = {norm(r): r for r in acc['Region_DHS']}
soc_lookup  = {norm(r): r for r in soc['Region_DHS']}

# Map GEO_REGION to DHS regions
# Some GeoJSON region names differ from DHS (e.g. "GREATER ACCRA" vs "Greater Accra")
# Strategy: normalised exact match first; manual overrides for mismatches
GEO_REGION_TO_DHS = {
    # Will be populated from exact matches; manual overrides below
}

def resolve_dhs_region(geo_reg, lookup):
    key = norm(geo_reg)
    if key in lookup:
        return lookup[key]
    # Try partial matches (first word)
    for dhs_key, dhs_val in lookup.items():
        if key.split()[0] == dhs_key.split()[0]:
            return dhs_val
    return None

dm['DHS_Region_NHIS'] = dm['GEO_REGION'].apply(
    lambda r: resolve_dhs_region(r, nhis_lookup) if pd.notna(r) else None)
dm['DHS_Region_Acc']  = dm['GEO_REGION'].apply(
    lambda r: resolve_dhs_region(r, acc_lookup)  if pd.notna(r) else None)
dm['DHS_Region_Soc']  = dm['GEO_REGION'].apply(
    lambda r: resolve_dhs_region(r, soc_lookup)  if pd.notna(r) else None)

# For Guan (Has_Geometry=False, GEO_REGION='Oti'):
# Oti Region is a 2019 administrative creation; map to closest DHS equivalent
guan_mask = dm['Has_Geometry'] == False
dm.loc[guan_mask, 'GEO_REGION'] = 'Oti'

# Check match coverage
nhis_matched = dm['DHS_Region_NHIS'].notna().sum()
acc_matched  = dm['DHS_Region_Acc'].notna().sum()
log(f"  NHIS region matched: {nhis_matched}/261")
log(f"  Access region matched: {acc_matched}/261")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 -- Merge DHS 2019 NHIS coverage (regional validation)
# ─────────────────────────────────────────────────────────────────────────────
log("\n[STEP 3] Merging DHS 2019 NHIS coverage (regional -> district)")

nhis_merge = nhis[['Region_DHS', 'NHIS_Coverage_Women_2019_pct',
                    'NHIS_Coverage_Men_2019_pct',
                    'No_Insurance_Women_2022_pct']].copy()

dm = dm.merge(nhis_merge, left_on='DHS_Region_NHIS',
              right_on='Region_DHS', how='left')
dm.drop(columns=['Region_DHS'], errors='ignore', inplace=True)

merged_nhis = dm['NHIS_Coverage_Women_2019_pct'].notna().sum()
log(f"  Districts with NHIS 2019 coverage: {merged_nhis}/261")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 -- Merge DHS 2022 access barriers (regional -> district)
# ─────────────────────────────────────────────────────────────────────────────
log("\n[STEP 4] Merging DHS 2022 access barriers (regional -> district)")

acc_merge = acc[['Region_DHS', 'ANC_Skilled_pct',
                  'Home_Delivery_pct', 'Facility_Delivery_pct']].copy()

dm = dm.merge(acc_merge, left_on='DHS_Region_Acc',
              right_on='Region_DHS', how='left')
dm.drop(columns=['Region_DHS'], errors='ignore', inplace=True)

merged_acc = dm['ANC_Skilled_pct'].notna().sum()
log(f"  Districts with access data: {merged_acc}/261")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 -- Merge DHS 2022 socioeconomic indicators (regional -> district)
# ─────────────────────────────────────────────────────────────────────────────
log("\n[STEP 5] Merging DHS 2022 socioeconomic indicators (regional -> district)")

soc_merge = soc[['Region_DHS', 'Female_Literate_pct',
                  'Female_No_Edu_pct', 'Female_SecPlus_pct']].copy()

dm = dm.merge(soc_merge, left_on='DHS_Region_Soc',
              right_on='Region_DHS', how='left')
dm.drop(columns=['Region_DHS'], errors='ignore', inplace=True)

merged_soc = dm['Female_Literate_pct'].notna().sum()
log(f"  Districts with socioeconomic data: {merged_soc}/261")

# Fill any unmatched regional values using region mean (Guan -> Oti region mean)
REG_FILL_COLS = ['NHIS_Coverage_Women_2019_pct', 'NHIS_Coverage_Men_2019_pct',
                 'No_Insurance_Women_2022_pct', 'ANC_Skilled_pct',
                 'Home_Delivery_pct', 'Facility_Delivery_pct',
                 'Female_Literate_pct', 'Female_No_Edu_pct', 'Female_SecPlus_pct']

for col in REG_FILL_COLS:
    n_miss = dm[col].isna().sum()
    if n_miss > 0:
        # Use GEO_REGION mean (for Guan: Oti region; but Guan in GEO_REGION='Oti'
        # which may not match — use overall mean as fallback)
        region_mean = dm.groupby('GEO_REGION')[col].transform('mean')
        overall_mean = dm[col].mean()
        dm[col] = dm[col].fillna(region_mean).fillna(overall_mean)
        log(f"  Filled {n_miss} missing in {col}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 -- Compute derived binary/categorical variables
# ─────────────────────────────────────────────────────────────────────────────
log("\n[STEP 6] Computing derived analytical variables")

# Binary outcome: high uninsurance (above national median)
median_unins = dm['Uninsurance_Rate_pct'].median()
dm['High_Uninsurance'] = (dm['Uninsurance_Rate_pct'] > median_unins).astype(int)
log(f"  National median Uninsurance_Rate_pct: {median_unins:.1f}%")
log(f"  High_Uninsurance=1 districts: {dm['High_Uninsurance'].sum()} / 261")

# Catastrophic spending proxy: high poverty AND high uninsurance
dm['Catastrophic_Proxy'] = (
    (dm['Poverty_Incidence_pct'] > 40) &
    (dm['Uninsurance_Rate_pct'] > 50)
).astype(int)
log(f"  Catastrophic_Proxy=1 (Poverty>40 AND Uninsurance>50): "
    f"{dm['Catastrophic_Proxy'].sum()} districts")

# Ecological zone classification based on GEO_REGION
NORTHERN_REGIONS = {'UPPER EAST', 'UPPER WEST', 'NORTH EAST', 'NORTHERN EAST', 'NORTHERN',
                    'SAVANNAH', 'OTI'}
MIDDLE_REGIONS   = {'BONO', 'BONO EAST', 'AHAFO', 'ASHANTI', 'EASTERN',
                    'VOLTA', 'BRONG-AHAFO'}
SOUTHERN_REGIONS = {'GREATER ACCRA', 'CENTRAL', 'WESTERN', 'WESTERN NORTH'}

def classify_zone(geo_region):
    r = str(geo_region).upper().strip()
    if r in NORTHERN_REGIONS:
        return 'Northern Belt'
    if any(r.startswith(m.split()[0]) for m in MIDDLE_REGIONS):
        return 'Middle Belt'
    if any(r.startswith(s.split()[0]) for s in SOUTHERN_REGIONS):
        return 'Southern Belt'
    return 'Other'

dm['Ecological_Zone'] = dm['GEO_REGION'].apply(classify_zone)
zone_counts = dm['Ecological_Zone'].value_counts()
log(f"  Ecological zones: {zone_counts.to_dict()}")

# Uninsurance quartile (for visualisation)
dm['Uninsurance_Quartile'] = pd.qcut(
    dm['Uninsurance_Rate_pct'], q=4,
    labels=['Q1 (lowest)', 'Q2', 'Q3', 'Q4 (highest)'])

# DHS 2019 did not record men's NHIS by region -- all NaN; drop to avoid imputation artefact
dm.drop(columns=['NHIS_Coverage_Men_2019_pct'], errors='ignore', inplace=True)
log("  [NOTE] NHIS_Coverage_Men_2019_pct dropped -- all NaN in DHS 2019 regional data")
log(f"  Total shape after all merges: {dm.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 -- Build spatial weights matrix (Queen contiguity, 260 districts)
# ─────────────────────────────────────────────────────────────────────────────
log("\n[STEP 7] Building Queen contiguity spatial weights (n=260)")

try:
    import libpysal
    from libpysal.weights import Queen

    # Load GeoJSON as GeoDataFrame
    import geopandas as gpd
    gdf = gpd.read_file(os.path.join(RAW, 'Ghana_New_260_District.geojson'))
    log(f"  GeoDataFrame loaded: {gdf.shape}")

    # Build Queen weights
    w = Queen.from_dataframe(gdf)
    w.transform = 'r'  # row-standardise
    log(f"  Queen weights built: {w.n} units, mean neighbours = {w.mean_neighbors:.2f}")
    log(f"  Islands (no neighbours): {len(w.islands)}")

    # Save
    w.to_file(os.path.join(PROC, 'spatial_weights_queen_260.gwt'))
    log("  Saved -> data/processed/spatial_weights_queen_260.gwt")

    # Also save as pickle for Python reuse
    with open(os.path.join(PROC, 'spatial_weights_queen_260.pkl'), 'wb') as f:
        pickle.dump(w, f)
    log("  Saved -> data/processed/spatial_weights_queen_260.pkl")

    WEIGHTS_OK = True

except Exception as e:
    log(f"  WARNING: Spatial weights build failed: {e}")
    log("  -> Install libpysal + geopandas: pip install libpysal geopandas --break-system-packages")
    WEIGHTS_OK = False

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 -- Generate Table 1 (descriptive statistics, N=261)
# ─────────────────────────────────────────────────────────────────────────────
log("\n[STEP 8] Generating Table 1 — descriptive statistics by ecological zone")

TABLE1_COLS = {
    'Uninsurance_Rate_pct':            'Uninsurance rate (%)',
    'Poverty_Incidence_pct':           'Poverty incidence (%)',
    'Poverty_Intensity_pct':           'Poverty intensity (%)',
    'Illiteracy_Rate_pct':             'Illiteracy rate (%)',
    'Youth_Dep_Ratio':                 'Youth dependency ratio',
    'Employment_Rate_pct':             'Employment rate (%)',
    'NHIS_Coverage_Women_2019_pct':    'NHIS coverage — women, 2019 (%)',
    'ANC_Skilled_pct':                 'ANC skilled provider coverage (%)',
    'Home_Delivery_pct':               'Home delivery (%)',
    'Female_Literate_pct':             'Female literacy (%)',
    'Female_SecPlus_pct':              'Female secondary+ education (%)',
    'High_Uninsurance':                'High uninsurance districts (n, %)',
    'Catastrophic_Proxy':              'Catastrophic spending proxy (n, %)',
}

zones = ['Overall'] + sorted(dm['Ecological_Zone'].unique().tolist())
table1_rows = []

for col, label in TABLE1_COLS.items():
    row = {'Variable': label}
    for zone in zones:
        sub = dm if zone == 'Overall' else dm[dm['Ecological_Zone'] == zone]
        n = len(sub)
        vals = sub[col].dropna()
        if col in ('High_Uninsurance', 'Catastrophic_Proxy'):
            cnt = int(vals.sum())
            pct = cnt / n * 100
            row[zone] = f"{cnt} ({pct:.1f}%)"
        else:
            row[zone] = f"{vals.mean():.1f} ± {vals.std():.1f}"
    row['N'] = int(dm[col].notna().sum())
    table1_rows.append(row)

table1 = pd.DataFrame(table1_rows)
table1.to_csv(os.path.join(PROC, 'table1_descriptive.csv'), index=False)
log(f"  Table 1 shape: {table1.shape}")
log("  Saved -> data/processed/table1_descriptive.csv")

# Print Table 1 summary
log("\n  TABLE 1 PREVIEW (Overall column):")
for _, row in table1.iterrows():
    log(f"    {row['Variable'][:45]:<45}  Overall: {row['Overall']}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 -- Final missing data check
# ─────────────────────────────────────────────────────────────────────────────
log("\n[STEP 9] Final missing data audit")

ANALYTICAL_COLS = [
    'Uninsurance_Rate_pct', 'Poverty_Incidence_pct', 'Poverty_Intensity_pct',
    'Illiteracy_Rate_pct', 'Youth_Dep_Ratio', 'Employment_Rate_pct',
    'NHIS_Coverage_Women_2019_pct', 'ANC_Skilled_pct', 'Home_Delivery_pct',
    'Female_Literate_pct', 'Female_SecPlus_pct',
    'High_Uninsurance', 'Catastrophic_Proxy', 'Ecological_Zone', 'Has_Geometry'
]

missing = dm[ANALYTICAL_COLS].isna().sum()
any_missing = missing[missing > 0]
if len(any_missing) == 0:
    log("  Missing data: NONE in all analytical columns")
else:
    log(f"  Missing data flagged:")
    for col, n in any_missing.items():
        log(f"    {col}: {n}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 -- Save final analytical dataset
# ─────────────────────────────────────────────────────────────────────────────
log("\n[STEP 10] Saving district_master_261_analytical.csv")

# Drop internal crosswalk columns
SAVE_COLS = [c for c in dm.columns if c not in
             ('DHS_Region_NHIS', 'DHS_Region_Acc', 'DHS_Region_Soc', '_KEY')]
dm_save = dm[SAVE_COLS].copy()

# Cast Uninsurance_Quartile to string for CSV compatibility
dm_save['Uninsurance_Quartile'] = dm_save['Uninsurance_Quartile'].astype(str)

dm_save.to_csv(os.path.join(PROC, 'district_master_261_analytical.csv'), index=False)
log(f"  Saved -> data/processed/district_master_261_analytical.csv")
log(f"  Final shape: {dm_save.shape}")
log(f"  Columns ({dm_save.shape[1]}): {list(dm_save.columns)}")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL -- Summary
# ─────────────────────────────────────────────────────────────────────────────
log("\n" + "=" * 70)
log("PHASE 2 DATA WRANGLING COMPLETE")
log("=" * 70)
log(f"  Primary dataset: district_master_261_analytical.csv "
    f"({dm_save.shape[0]} x {dm_save.shape[1]})")
log(f"  Spatial districts (Has_Geometry=True):  {dm_save['Has_Geometry'].sum()}")
log(f"  Non-spatial (Has_Geometry=False):       {(~dm_save['Has_Geometry']).sum()}")
log(f"  High_Uninsurance districts:             {dm_save['High_Uninsurance'].sum()}")
log(f"  Catastrophic_Proxy districts:           {dm_save['Catastrophic_Proxy'].sum()}")
log(f"  Spatial weights:                        {'Built OK' if WEIGHTS_OK else 'FAILED -- install libpysal'}")
log(f"  Table 1:                                table1_descriptive.csv")

with open(os.path.join(PROC, 'wrangling_audit.txt'), 'w') as fh:
    fh.write('\n'.join(audit))

print("\nWRANGLING COMPLETE -- 2026-05-28")
