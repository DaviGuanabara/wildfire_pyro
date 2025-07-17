
## 🧠 Project Overview

This project aims to build a predictive model that relates **daily microclimate variables** to **Leaf Wetness Measurement (LWM)** — a key indicator for fungal and bacterial disease risk in vineyards. Data was collected from the [Iowa Environmental Mesonet (IEM)](https://mesonet.agron.iastate.edu), focusing on **vineyard weather stations** across the state of Iowa (USA).

---

## 📍 Motivation

Leaf wetness is a critical environmental condition for the development of plant diseases. Accurate prediction enables early intervention in vineyard disease management and irrigation planning.

---

## 🌐 Data Source

- **Main portal:** [IEM AgClimate Daily Tool](https://mesonet.agron.iastate.edu/agclimate/hist/daily.php)
- **Station network:** [ISUSM – Iowa State University Soil Moisture Network](https://mesonet.agron.iastate.edu/sites/networks.php?network=ISUSM)

---

## 📅 Time Span

- **Start Date:** 2020-01-01  
- **End Date:** 2025-06-30

---

## 🛰️ Selected Stations

| Station Name                        | Code  | Type       | Used? |
|------------------------------------|-------|------------|-------|
| Ames – Horticulture Vineyard       | AHTI4 | Vineyard   | ✅ Yes |
| Bankston – Park Farm Winery        | BNKI4 | Vineyard   | ✅ Yes |
| Oskaloosa – Tassel Ridge           | OSTI4 | Vineyard   | ✅ Yes |
| Glenwood – Blackwing Vineyard      | GVNI4 | Vineyard   | ✅ Yes |
| Ames – Finch Farm (central station)| AMFI4 | Central    | ❌ Removed (not vineyard-related) |

---

## 🌡️ Collected Variables

### 📌 General climatic variables (used as features):

- `high`: Daily max temperature [°F]
- `low`: Daily min temperature [°F]
- `rh`, `rh_min`, `rh_max`: Relative humidity [%]
- `solar_mj`: Solar radiation [MJ/m²]
- `precip`: Precipitation [inches]
- `speed`: Average wind speed [mph]
- `gust`: Wind gust [mph]
- `et`: Reference evapotranspiration [inches]
- `bpres_avg`: Barometric pressure [mb]

### 🍇 Vineyard-specific variables (LWM – used as target):

- `lwmv_1`, `lwmv_2`: Leaf wetness measurement (sensor 1 and 2)
- `lwmdry_1_tot`, `lwmdry_2_tot`: Total dry hours
- `lwmwet_1_tot`, `lwmwet_2_tot`: Total wet hours
- `lwmcon_1_tot`, `lwmcon_2_tot`: Consecutive wet hours

### 🌱 SoilVue (removed):

Soil temperature and moisture data were discarded due to lack of coverage across vineyard stations.

---

## ⚙️ Preprocessing Summary

1. ✅ Merged data with station metadata (name, coordinates, elevation)
2. ✅ Renamed and standardized columns
3. ✅ Removed:
   - Central station (no vineyard data)
   - Soil-related columns (`sv_*`)
   - Columns with >50% missing values
   - `_f` quality flag columns (after inspection)
4. ✅ Dropped rows with missing target or essential features
5. ✅ Final dataset exported to `dataset_cleaned.xlsx`

---

## 🏷️ About `_f` Quality Flags

Each variable has an associated quality flag (suffix `_f`), used by IEM for internal quality control:

| Flag | Meaning                                 |
|------|------------------------------------------|
| `E`  | Estimated (value inferred by IEM)        |
| `M`  | Missing                                  |
| `T`  | Trace amount (extremely small)           |
| `A`  | Accumulated across multiple days         |

➡️ Values marked as `E` (Estimated) were retained in the cleaned dataset. Others were discarded.

---

## 🧪 Modeling Strategy

- **Task:** Regression — Predict LWM using daily weather conditions.
- **Target variable (initial):** `lwmv_1`
- **Baseline models:** Linear regression, Random Forest
- **Future models:** Attention-based neural networks, spatiotemporal modeling

---

## ✅ Recommended Variables (Phase 1)

| Variable        | Include? | Rationale                                          |
|----------------|----------|----------------------------------------------------|
| `high`, `low`  | ✅ Yes    | Core predictors (temperature range)                |
| `solar_mj`     | ✅ Yes    | Key for evaporation and drying                    |
| `precip`       | ✅ Yes    | Direct impact on leaf wetness                     |
| `rh`, `rh_min` | ✅ Yes    | Controls moisture retention                       |
| `speed`        | ⚖️ Optional | Wind contributes to drying                     |
| `et`           | ⚖️ Optional | Derived metric; useful but redundant             |
| `bpres_avg`    | ⚖️ Optional | Indirectly relates to precipitation patterns     |

---

## 📤 Current Output

- ✅ Clean dataset: `dataset_cleaned.xlsx`
- ✅ Feature/target schema defined
- 🚧 Modeling pipeline in development
- 🧾 This document serves as a technical reference

