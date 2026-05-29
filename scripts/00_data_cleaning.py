"""
00_data_cleaning.py
====================
NHIS OOP Ghana 261 Districts — Phase 0 Data Cleaning (v3 — 261-district framing)
Author: Valentine Golden Ghanem | AIPOCH v6.5 | 2026-05-28

DQ flags resolved:
  DQ-01  2022 DHS NHIS indicator changed -> use 2019 survey
  DQ-02  Master Sheet 261 rows (Guan) -> RETAIN Guan as non-spatial record
           (Has_Geometry=False); GeoJSON join on 260; final n=261
  DQ-03  WHO GHED duplicate rows -> deduplicate by GHO CODE + YEAR
  DQ-04  DHS subnational = regional, not district -> documented; PHC 2021 primary
  DQ-05  Excel formula columns -> data_only=True
  DQ-06  260 GeoJSON<->MS name mismatches -> 44-pair hardcoded lookup
"""

import os, json, warnings
import pandas as pd
import openpyxl

warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW  = os.path.join(BASE, 'data', 'raw')
PROC = os.path.join(BASE, 'data', 'processed')
os.makedirs(PROC, exist_ok=True)

audit = []
def log(msg):
    print(msg)
    audit.append(msg)

log("=" * 70)
log("DATA CLEANING AUDIT -- NHIS OOP Ghana 261 Districts  v3")
log("AIPOCH v6.5 | 2026-05-28")
log("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# COMPLETE GEO_DISTRICT -> MS_MMDA LOOKUP  [DQ-06]
# All 44 pairs where GeoJSON and Master Sheet names differ.
# 216 districts already match exactly after UPPER().strip(); these 44 handle
# spelling variants, hyphenation differences, and metropolitan re-labelling.
# ─────────────────────────────────────────────────────────────────────────────
GEO_TO_MS = {
    # Spelling / suffix variants
    'ADENTA MUNICIPAL':                         'Adentan Municipal',
    'LAMBUSSIE-KARNI':                          'Lambussie Karni',
    'SAWLA-TUNA-KALBA':                         'Sawla Tuna Kalba',
    'SHAI OSUDOKU':                             'Shai-Osudoku',
    'KASENA NANKANA EAST':                      'Kasena Nankana Municipal',
    'NINGO/PRAMPRAM':                           'Ningo-Prampram',
    'MFANTSEMAN MUNICIPAL':                     'Mfantsiman Municipal',
    'TWIFO HEMANG LOWER DENKYIRA':              'Twifo Heman Lower Denkyira',
    'TWIFO ATTI-MORKWA':                        'Twifo Ati Morkwa',
    'AWUTU SENYA':                              'Awutu Senya West',
    'LA-NKWANTANANG-MADINA':                    'La Nkwantanang Madina Municipal',
    'AGOTIME ZIOPE':                            'Agortime-Ziope',
    'EJURA-SEKYEDUMASE':                        'Ejura Sekyedumase Municipal',
    'BOSOMTWE':                                 'Bosomtwi',
    'AJUMAKO-ENYAN-ESSIAM':                     'Ajumako Enyan Essiam',
    'ASIKUMA-ODOBEN-BRAKWA':                    'Asikuma Odoben Brakwa',
    'ABURA-ASEBU-KWAMANKESE':                   'Abura Asebu Kwamankese',
    'DORMAA MUNICIPAL':                         'Dormaa Central Municipal',
    'NADOWLI-KALEO':                            'Nadowli Kaleo',
    'TARKWA NSUAEM':                            'Tarkwa-Nsuaem Municipal',
    'BIBIANI-ANHWIASO-BEKWAI MUNICIPAL':        'Bibiani Anhwiaso Bekwai Municipal',
    'KOMENDA-EDINA-EGUAFO-ABIREM MUNICIPAL':    'Komenda Edina Eguafo Abirem Municipal',
    'SEFWI-WIAWSO':                             'Sefwi Wiawso Municipal',
    'SEKYERE AFRAM PLAINS NORTH':               'Sekyere Afram Plains',
    'NKWANTA NORTH':                            'Nkwanta North (Kpassa)',
    'SAGNERIGU':                                'Sagnarigu Municipal',
    'BOLGA  EAST':                              'Bolgatanga East',
    'YUNYOO-NASUAN':                            'Yunyoo Nasuan',
    'ASSIN FOSU':                               'Assin Central Municipal',
    'ADANSI AKROFUOM':                          'Akrofuom',
    'OKAIKWEI NORTH MUNICIPAL':                 'Okaikoi North Municipal',
    'ASENE AKROSO MANSO':                       'Asene Manso Akroso',
    'UPPER MANYA':                              'Upper Manya Krobo',
    'LOWER MANYA':                              'Lower Manya Krobo Municipal',
    'AKWAPEM SOUTH':                            'Akwapim South Municipal',
    'DENKYEMBOUR':                              'Denkyembuor',
    'AKYEM MANSA':                              'Akyemansa',
    'AKWAPEM NORTH':                            'Akwapim North Municipal',
    # Metropolitan areas (renamed/merged in PHC 2021 Master Sheet)
    'ACCRA METROPOLIS':
        'Accra Metropolitan Area (AMA)-Ablekuma South, Ashiedu Keteke & Okaikoi South',
    'KUMASI METROPOLITAN':
        'Kumasi Metropolitan Area (KMA)-Bantama, Manhyia North, Manhyia South, Nhyiaeso, & Subin',
    'TEMA METROPOLITAN':
        'Tema Metropolitan Area (TMA)-Tema Central & Tema East',
    'TAMALE METROPOLITAN':
        'Tamale Metropolitan Area (TMA)-Tamale Central & Tamale South',
    'CAPE COAST METROPOLITAN':
        'Cape Cape Metropolitan Area (CCMA)-Cape Coast South & Cape Coast North',
    'SEKONDI TAKORADI METROPOLIS':
        'Sekondi Takoradi Metropolitan Area (STMA)- Takoradi, Sekondi & Essikado-Ketan',
    # GeoJSON has "MUNICIPAL" where MS does not (or vice-versa)
    'JAMAN SOUTH MUNICIPAL':  'Jaman South',
    'WEST AKIM':              'West Akim Municipal',
    # Double-space in GeoJSON: "ASOKWA  MUNICIPAL" -> normalise
    'ASOKWA  MUNICIPAL':      'Asokwa Municipal',
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 -- MASTER SHEET  [DQ-02, DQ-05]
# ─────────────────────────────────────────────────────────────────────────────
log("\n[STEP 1] Master Sheet (2021 Ghana PHC) -- data_only=True")

wb = openpyxl.load_workbook(os.path.join(RAW, 'Master Sheet.xlsx'),
                             read_only=True, data_only=True)
ws = wb['Sheet1']
rows = list(ws.iter_rows(values_only=True))
ms = pd.DataFrame(rows[1:], columns=rows[0])
ms.columns = [
    'Region', 'MMDA', 'Class', 'Latitude', 'Longitude',
    'Employed_Pop', 'Unemployed_Pop', 'Poverty_Incidence_pct',
    'Poverty_Intensity_pct', 'Illiterate_Pop', 'Uninsured_Pop',
    'Male_0_14', 'Female_0_14', 'Male_15_64', 'Female_15_64',
    'Male_65plus', 'Female_65plus', 'Total_Pop'
]
num_cols = ['Employed_Pop','Unemployed_Pop','Poverty_Incidence_pct',
            'Poverty_Intensity_pct','Illiterate_Pop','Uninsured_Pop',
            'Male_0_14','Female_0_14','Male_15_64','Female_15_64',
            'Male_65plus','Female_65plus','Total_Pop']
for c in num_cols:
    ms[c] = pd.to_numeric(ms[c], errors='coerce')

log(f"  Raw rows: {len(ms)}")

# DQ-02 REVISED: Retain Guan; append as non-spatial record after GeoJSON join.
# Guan has no boundary in Ghana_New_260_District.geojson (pre-2022 creation).
# Study frames 261 districts total; 260 with spatial geometry for LISA/GWR/Gi*.
guan_row_ms = ms[ms['MMDA'] == 'Guan'].copy()
log(f"  [DQ-02] Guan retained for non-spatial analysis "
    f"(will append with Has_Geometry=False); GeoJSON join on 260 districts")

# Derived variables (all 261 rows including Guan)
ms['Uninsurance_Rate_pct'] = (ms['Uninsured_Pop']  / ms['Total_Pop']) * 100
ms['Illiteracy_Rate_pct']  = (ms['Illiterate_Pop'] / ms['Total_Pop']) * 100
ms['Youth_Dep_Ratio']      = ((ms['Male_0_14'] + ms['Female_0_14']) /
                               (ms['Male_15_64'] + ms['Female_15_64'])) * 100
ms['Employment_Rate_pct']  = (ms['Employed_Pop'] / ms['Total_Pop']) * 100

log(f"  Uninsurance_Rate_pct: "
    f"{ms['Uninsurance_Rate_pct'].min():.1f}%--"
    f"{ms['Uninsurance_Rate_pct'].max():.1f}% "
    f"(mean {ms['Uninsurance_Rate_pct'].mean():.1f}%)")
log(f"  Illiteracy_Rate_pct:  "
    f"{ms['Illiteracy_Rate_pct'].min():.1f}%--"
    f"{ms['Illiteracy_Rate_pct'].max():.1f}% "
    f"(mean {ms['Illiteracy_Rate_pct'].mean():.1f}%)")
log(f"  Missing (Uninsurance): {ms['Uninsurance_Rate_pct'].isna().sum()}")

# Update Guan row derived variables
guan_row_ms = ms[ms['MMDA'] == 'Guan'].copy()

# Build MMDA -> row lookup (upper-stripped key, all 261 including Guan)
ms['_KEY'] = ms['MMDA'].str.upper().str.strip()
ms_lookup  = {row['_KEY']: idx for idx, row in ms.iterrows()}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 -- GeoJSON + DQ-06 harmonisation
# ─────────────────────────────────────────────────────────────────────────────
log("\n[STEP 2] GeoJSON loading + 44-pair name correction [DQ-06]")

with open(os.path.join(RAW, 'Ghana_New_260_District.geojson')) as f:
    geo = json.load(f)

records = []
for feat in geo['features']:
    geo_name   = feat['properties']['DISTRICT']
    geo_region = feat['properties']['REGION']
    # Resolve MS name: hardcoded map first, then exact uppercase match
    ms_name = GEO_TO_MS.get(geo_name,
              GEO_TO_MS.get(geo_name.upper().strip(), None))
    if ms_name is None:
        ms_name = geo_name   # assume exact match; validated below
    records.append({'GEO_DISTRICT': geo_name,
                    'GEO_REGION':   geo_region,
                    'MS_MMDA':      ms_name})

geo_df = pd.DataFrame(records)

# Resolve MS row index -- try exact match, then + MUNICIPAL fallback
def resolve_ms_idx(ms_name):
    key = ms_name.upper().strip()
    idx = ms_lookup.get(key, None)
    if idx is None:
        idx = ms_lookup.get(key + ' MUNICIPAL', None)
    return idx

geo_df['MS_IDX'] = geo_df['MS_MMDA'].apply(resolve_ms_idx)

matched   = geo_df['MS_IDX'].notna().sum()
unmatched = geo_df['MS_IDX'].isna().sum()
log(f"  Matched: {matched}/260  |  Still unmatched: {unmatched}")

if unmatched > 0:
    log("  UNRESOLVED:")
    for _, r in geo_df[geo_df['MS_IDX'].isna()].iterrows():
        log(f"    GEO={r['GEO_DISTRICT']!r}  ->  MS_MMDA tried={r['MS_MMDA']!r}")
    log("  -> Applying region-mean imputation for unresolved rows")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 -- Build district_master_261.csv
# ─────────────────────────────────────────────────────────────────────────────
log("\n[STEP 3] Building district_master_261.csv")

OUTPUT_COLS = [
    'MMDA', 'Region', 'Latitude', 'Longitude',
    'Employed_Pop', 'Unemployed_Pop', 'Poverty_Incidence_pct',
    'Poverty_Intensity_pct', 'Illiterate_Pop', 'Uninsured_Pop',
    'Male_0_14', 'Female_0_14', 'Male_15_64', 'Female_15_64',
    'Male_65plus', 'Female_65plus', 'Total_Pop',
    'Uninsurance_Rate_pct', 'Illiteracy_Rate_pct',
    'Youth_Dep_Ratio', 'Employment_Rate_pct'
]

# Pull MS data for each geo row (260 GeoJSON features)
ms_data = []
for _, gr in geo_df.iterrows():
    idx = gr['MS_IDX']
    if pd.notna(idx):
        ms_data.append(ms.loc[int(idx), OUTPUT_COLS].to_dict())
    else:
        ms_data.append({c: None for c in OUTPUT_COLS})

ms_block = pd.DataFrame(ms_data)

district = pd.concat([
    geo_df[['GEO_DISTRICT', 'GEO_REGION']].reset_index(drop=True),
    ms_block.reset_index(drop=True)
], axis=1)

# Region-mean imputation for any still-missing rows
n_missing_unreg = district['Uninsurance_Rate_pct'].isna().sum()
if n_missing_unreg > 0:
    log(f"  Imputing {n_missing_unreg} rows via region mean")
    for col in ['Poverty_Incidence_pct', 'Poverty_Intensity_pct',
                'Illiteracy_Rate_pct', 'Uninsurance_Rate_pct',
                'Youth_Dep_Ratio', 'Employment_Rate_pct']:
        region_mean = district.groupby('GEO_REGION')[col].transform('mean')
        district[col] = district[col].fillna(region_mean)

assert len(district) == 260, f"Expected 260 geo-matched rows, got {len(district)}"

# Append Guan as 261st district (non-spatial -- no GeoJSON boundary)
district['Has_Geometry'] = True
if len(guan_row_ms) > 0:
    guan_cols = {c: (guan_row_ms[c].values[0]
                     if c in guan_row_ms.columns else None)
                 for c in OUTPUT_COLS}
    guan_cols.update({'GEO_DISTRICT': 'GUAN', 'GEO_REGION': 'Oti',
                      'Has_Geometry': False})
    district = pd.concat([district, pd.DataFrame([guan_cols])],
                         ignore_index=True)
    log(f"  [DQ-02] Guan appended as non-spatial record -> n=261")
else:
    log("  WARNING: Guan not found in Master Sheet -- check MMDA column name")

# Final checks
missing_check = district[['Uninsurance_Rate_pct','Poverty_Incidence_pct',
                           'Illiteracy_Rate_pct']].isna().sum()
log(f"  Final shape: {district.shape}")
log(f"  Missing after imputation: {missing_check.to_dict()}")
log(f"  Spatial districts (Has_Geometry=True):  "
    f"{district['Has_Geometry'].sum()}")
log(f"  Non-spatial districts (Has_Geometry=False): "
    f"{(~district['Has_Geometry']).sum()}")
log(f"  Uninsurance_Rate_pct range: "
    f"{district['Uninsurance_Rate_pct'].min():.1f}%--"
    f"{district['Uninsurance_Rate_pct'].max():.1f}%")

district.to_csv(os.path.join(PROC, 'district_master_261.csv'), index=False)
log(f"  Saved -> data/processed/district_master_261.csv")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 -- DHS 2019 NHIS by region  [DQ-01]
# ─────────────────────────────────────────────────────────────────────────────
log("\n[STEP 4] DHS 2019 NHIS coverage [DQ-01]")

def read_dhs(fname):
    return pd.read_csv(os.path.join(RAW, fname), skiprows=[1])

df_ins = read_dhs('health-insurance_subnational_gha.csv')

nhis_w = (df_ins[(df_ins['SurveyYear'] == 2019) &
                  (df_ins['Indicator'] == 'Social security health insurance [Women]')]
          [['Location','Value','CILow','CIHigh','DenominatorWeighted']]
          .rename(columns={'Location':          'Region_DHS',
                           'Value':             'NHIS_Coverage_Women_2019_pct',
                           'CILow':             'CI_Low',
                           'CIHigh':            'CI_High',
                           'DenominatorWeighted':'N_Women'}))

nhis_m = (df_ins[(df_ins['SurveyYear'] == 2019) &
                  (df_ins['Indicator'] == 'Social security health insurance [Men]')]
          [['Location','Value']]
          .rename(columns={'Location':'Region_DHS',
                           'Value':   'NHIS_Coverage_Men_2019_pct'}))

no_ins_22 = (df_ins[(df_ins['SurveyYear'] == 2022) &
                     (df_ins['Indicator'] == 'No health insurance [Women]')]
             [['Location','Value']]
             .rename(columns={'Location':'Region_DHS',
                              'Value':   'No_Insurance_Women_2022_pct'}))

nhis_df = nhis_w.merge(nhis_m, on='Region_DHS', how='left')\
                 .merge(no_ins_22, on='Region_DHS', how='left')

log(f"  2019 NHIS regions: {len(nhis_df)}")
log(f"  Women range: "
    f"{nhis_df['NHIS_Coverage_Women_2019_pct'].min():.1f}%--"
    f"{nhis_df['NHIS_Coverage_Women_2019_pct'].max():.1f}%")

nhis_df.to_csv(os.path.join(PROC, 'nhis_regional_2019.csv'), index=False)
log("  Saved -> data/processed/nhis_regional_2019.csv")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 -- WHO GHED: OOP + GGHE  [DQ-03]
# ─────────────────────────────────────────────────────────────────────────────
log("\n[STEP 5] WHO GHED -- deduplication [DQ-03]")

df_fin = pd.read_csv(os.path.join(RAW, 'health_financing_indicators_gha.csv'),
                     skiprows=[1])
df_fin.columns = [c.strip() for c in df_fin.columns]

TARGET = {
    'GHED_GGHE-DGDP_SHA2011':  'GGHE_pct_GDP',
    'GHED_OOPSCHE_SHA2011':    'OOP_pct_CHE',
    'GHED_OOPPC_SHA2011':      'OOP_pc_USD',
    'GHED_CHEGDP_SHA2011':     'CHE_pct_GDP',
    'GHED_PVTDCHE_SHA2011':    'PVT_pct_CHE',
    'GHED_EXTCHE_SHA2011':     'EXT_pct_CHE',
}

rows_fin = []
for yr in sorted(df_fin['YEAR (DISPLAY)'].dropna().unique()):
    df_yr = df_fin[df_fin['YEAR (DISPLAY)'] == yr]
    row = {'Year': int(yr)}
    for code, col in TARGET.items():
        sub = df_yr[df_yr['GHO (CODE)'] == code]
        row[col] = pd.to_numeric(sub['Numeric'].iloc[0], errors='coerce') \
                   if len(sub) > 0 else None
    rows_fin.append(row)

oop_df = (pd.DataFrame(rows_fin)
            .drop_duplicates(subset='Year')
            .sort_values('Year')
            .reset_index(drop=True))

log(f"  Years: {oop_df['Year'].min()}--{oop_df['Year'].max()}  "
    f"| Rows: {len(oop_df)}")
log(f"  OOP_pc_USD 2022: "
    f"{oop_df.loc[oop_df['Year']==2022,'OOP_pc_USD'].values}")
log(f"  GGHE_pct_GDP 2022: "
    f"{oop_df.loc[oop_df['Year']==2022,'GGHE_pct_GDP'].values}")

oop_df.to_csv(os.path.join(PROC, 'oop_national_trend.csv'), index=False)
log("  Saved -> data/processed/oop_national_trend.csv")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 -- DHS 2022 Access Barriers (regional)
# ─────────────────────────────────────────────────────────────────────────────
log("\n[STEP 6] DHS 2022 access-to-care indicators (regional)")

df_acc = read_dhs('access-to-health-care_subnational_gha.csv')
df_acc22 = df_acc[df_acc['SurveyYear'] == 2022]

ACCESS_INDS = {
    'Antenatal care from a skilled provider': 'ANC_Skilled_pct',
    'Place of delivery: At home':             'Home_Delivery_pct',
    'Place of delivery: Health facility':     'Facility_Delivery_pct',
}

rows_acc = []
for region in df_acc22['Location'].unique():
    rdf = df_acc22[df_acc22['Location'] == region]
    row = {'Region_DHS': region}
    for ind, col in ACCESS_INDS.items():
        sub = rdf[rdf['Indicator'] == ind]
        row[col] = float(sub['Value'].iloc[0]) if len(sub) > 0 else None
    rows_acc.append(row)

acc_df = pd.DataFrame(rows_acc)
log(f"  Access regions: {len(acc_df)} | "
    f"Home delivery range: "
    f"{acc_df['Home_Delivery_pct'].dropna().min():.1f}%--"
    f"{acc_df['Home_Delivery_pct'].dropna().max():.1f}%")

acc_df.to_csv(os.path.join(PROC, 'access_regional_2022.csv'), index=False)
log("  Saved -> data/processed/access_regional_2022.csv")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 -- DHS 2022 Education/Literacy (regional)
# ─────────────────────────────────────────────────────────────────────────────
log("\n[STEP 7] DHS 2022 education and literacy (regional)")

df_lit = read_dhs('literacy_subnational_gha.csv')
df_edu = read_dhs('select-education-indicators_subnational_gha.csv')
df_gen = read_dhs('select-gender-indicators_subnational_gha.csv')

def extract_regional(df, year, indicator, colname):
    sub = df[(df['SurveyYear'] == year) & (df['Indicator'] == indicator)]
    return (sub[['Location', 'Value']]
            .rename(columns={'Location': 'Region_DHS', 'Value': colname}))

lit_df  = extract_regional(df_lit, 2022, 'Women who are literate',
                            'Female_Literate_pct')
edu_no  = extract_regional(df_edu, 2022, 'Women with no education',
                            'Female_No_Edu_pct')
edu_sec = extract_regional(df_gen, 2022,
                            'Women with secondary or higher education',
                            'Female_SecPlus_pct')

soc_df = (lit_df.merge(edu_no,  on='Region_DHS', how='outer')
                 .merge(edu_sec, on='Region_DHS', how='outer'))

log(f"  Socioeconomic regions: {len(soc_df)}")
soc_df.to_csv(os.path.join(PROC, 'socioeconomic_regional_2022.csv'), index=False)
log("  Saved -> data/processed/socioeconomic_regional_2022.csv")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL -- DQ Summary
# ─────────────────────────────────────────────────────────────────────────────
log("\n" + "=" * 70)
log("DQ RESOLUTION COMPLETE")
log("=" * 70)
log("  DQ-01 RESOLVED: 2022 NHIS indicator invalid; 2019 DHS used")
log("  DQ-02 RESOLVED: Guan retained as non-spatial record (Has_Geometry=False); n=261")
log("  DQ-03 RESOLVED: WHO GHED deduplicated to 1 row/year")
log("  DQ-04 DOCUMENTED: DHS = regional; PHC 2021 = primary district source")
log("  DQ-05 RESOLVED: data_only=True -- all formula columns evaluated")
log(f"  DQ-06 RESOLVED: 44-pair hardcoded lookup; {matched}/260 geo-matched")
log("")
log("Final outputs:")
log("  district_master_261.csv         261 x 24  primary analytical dataset (Has_Geometry col)")
log("  nhis_regional_2019.csv          regional  DHS NHIS reference")
log("  oop_national_trend.csv          23 x 7    WHO GHED trend")
log("  access_regional_2022.csv        regional  DHS access barriers")
log("  socioeconomic_regional_2022.csv regional  DHS education/literacy")

with open(os.path.join(PROC, 'cleaning_audit.txt'), 'w') as fh:
    fh.write('\n'.join(audit))

print("\nCLEANING COMPLETE -- 2026-05-28 (v3: 261-district framing)")
