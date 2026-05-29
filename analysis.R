# analysis.R — NHIS OOP Ghana 261 Districts | AIPOCH v6.5
# Full spatial analysis pipeline: Steps 4-8 + spatial regression diagnostics
# Steps: (4) Queen weights | (5) Global Moran's I | (6) Bivariate LISA |
#         (7) Getis-Ord Gi* | (8) GWR spatially varying coefficients | SLM/SEM
# Requires: spdep, spatialreg, GWmodel, sf, dplyr
# Run from project root: Rscript analysis.R
# Author: Valentine Golden Ghanem | AIPOCH v6.5 | 2026-05-29

suppressPackageStartupMessages({
  library(spdep)
  library(spatialreg)
  library(sf)
  library(dplyr)
})

gwr_available <- requireNamespace("GWmodel", quietly = TRUE)
if (gwr_available) suppressPackageStartupMessages(library(GWmodel))

cat("=== NHIS OOP Ghana -- Full Spatial Analysis (Steps 4-8) ===\n\n")

# --- STEP 4: DATA LOAD & SPATIAL WEIGHTS (Queen contiguity, n=260) ---

df <- read.csv("data/processed/spatial_results.csv", stringsAsFactors = FALSE)
df_spatial <- df[df$Has_Geometry == "True", ]
cat(sprintf("[STEP 4] Spatial districts loaded: %d\n", nrow(df_spatial)))

if (!file.exists("data/raw/Ghana_New_260_District.geojson")) {
  stop("GeoJSON not found: data/raw/Ghana_New_260_District.geojson")
}
shp <- st_read("data/raw/Ghana_New_260_District.geojson", quiet = TRUE)
cat(sprintf("         GeoJSON polygons loaded: %d\n", nrow(shp)))

nb_queen <- poly2nb(shp, queen = TRUE)
lw       <- nb2listw(nb_queen, style = "W")
lw_b     <- nb2listw(nb_queen, style = "B")

cat(sprintf("         Queen contiguity: %d links | Mean neighbours: %.2f\n\n",
            sum(card(nb_queen)), mean(card(nb_queen))))

# --- STEP 5: GLOBAL MORAN'S I ---

cat("[STEP 5] Global Moran's I -- Uninsurance_Rate_pct\n")
z_unins <- scale(df_spatial$Uninsurance_Rate_pct)[, 1]
moran_u <- moran.test(z_unins, lw)
cat(sprintf("         Moran's I = %.4f | z = %.3f | p = %.4f\n",
            moran_u$estimate["Moran I statistic"],
            moran_u$statistic,
            moran_u$p.value))
cat(sprintf("         Interpretation: %s spatial autocorrelation\n\n",
            ifelse(moran_u$p.value < 0.05, "SIGNIFICANT positive", "Non-significant")))

# --- STEP 6: BIVARIATE LISA ---

cat("[STEP 6] Bivariate LISA (Uninsurance x Poverty; Uninsurance x Illiteracy)\n")

z_pov   <- scale(df_spatial$Poverty_Incidence_pct)[, 1]
z_illit <- scale(df_spatial$Illiteracy_Rate_pct)[, 1]

wz_pov   <- lag.listw(lw, z_pov)
wz_illit <- lag.listw(lw, z_illit)

bv_pov_cor   <- cor.test(z_unins, wz_pov)
bv_illit_cor <- cor.test(z_unins, wz_illit)

cat(sprintf("         Uninsurance x W*Poverty:    r = %.4f | p = %.4f\n",
            bv_pov_cor$estimate, bv_pov_cor$p.value))
cat(sprintf("         Uninsurance x W*Illiteracy: r = %.4f | p = %.4f\n",
            bv_illit_cor$estimate, bv_illit_cor$p.value))

bv_classify <- function(x_std, wy_std) {
  wy_sc <- as.numeric(scale(wy_std))
  ifelse(x_std > 0 & wy_sc > 0, "HH",
   ifelse(x_std < 0 & wy_sc < 0, "LL",
   ifelse(x_std > 0 & wy_sc < 0, "HL", "LH")))
}

bv_quad_pov   <- bv_classify(z_unins, wz_pov)
bv_quad_illit <- bv_classify(z_unins, wz_illit)

cat("\n         BV-LISA quadrants (Uninsurance x Poverty):\n")
print(table(bv_quad_pov))
cat("\n         BV-LISA quadrants (Uninsurance x Illiteracy):\n")
print(table(bv_quad_illit))
cat("\n")

# --- STEP 7: GETIS-ORD Gi* HOTSPOT DELINEATION ---

cat("[STEP 7] Getis-Ord Gi* -- Hotspot / Coldspot delineation\n")

gi_star <- localG_perm(df_spatial$Uninsurance_Rate_pct, lw_b,
                       nsim = 499, alternative = "two.sided")
gi_z <- as.numeric(gi_star)

internals <- attr(gi_star, "internals")
p_gi <- if (!is.null(internals) && "Pr(z != E(Gi))" %in% colnames(internals)) {
  internals[, "Pr(z != E(Gi))"]
} else {
  2 * pnorm(abs(gi_z), lower.tail = FALSE)
}

gi_class <- ifelse(gi_z >= 2.576 & p_gi < 0.01,  "HH_p001",
             ifelse(gi_z >= 1.960 & p_gi < 0.05,  "HH_p005",
             ifelse(gi_z <= -2.576 & p_gi < 0.01, "LL_p001",
             ifelse(gi_z <= -1.960 & p_gi < 0.05, "LL_p005", "NS"))))

cat(sprintf("         Hotspot districts (p<0.01): %d\n", sum(gi_class == "HH_p001")))
cat(sprintf("         Hotspot districts (p<0.05): %d\n",
            sum(gi_class %in% c("HH_p001","HH_p005"))))
cat(sprintf("         Coldspot districts (p<0.01): %d\n", sum(gi_class == "LL_p001")))
cat(sprintf("         Coldspot districts (p<0.05): %d\n",
            sum(gi_class %in% c("LL_p001","LL_p005"))))
cat(sprintf("         Not significant: %d\n\n", sum(gi_class == "NS")))

hot_idx <- order(gi_z, decreasing = TRUE)[1:5]
cat("         Top 5 hotspot districts:\n")
for (i in seq_along(hot_idx)) {
  cat(sprintf("           %d. %s (Gi* = %.3f)\n",
              i, df_spatial$GEO_DISTRICT[hot_idx[i]], gi_z[hot_idx[i]]))
}
cat("\n")

# --- STEP 8: GWR SPATIALLY VARYING COEFFICIENTS ---

cat("[STEP 8] Geographically Weighted Regression (GWR)\n")

if (!gwr_available) {
  cat("         GWmodel not installed -- skipping R GWR.\n")
  cat("         Python GWR results (scripts/02_spatial_analysis.py):\n")
  cat("           Mean local R-squared = 0.624 (range 0.239-0.895)\n")
  cat("           Mean beta Poverty = 0.625\n")
  cat("         To enable R GWR: install.packages('GWmodel')\n\n")
} else {
  sp_df <- as(shp, "Spatial")
  sp_df@data <- df_spatial[, c("Uninsurance_Rate_pct",
                                "Poverty_Incidence_pct",
                                "Illiteracy_Rate_pct",
                                "NHIS_Coverage_Women_2019_pct")]

  gwr_formula <- Uninsurance_Rate_pct ~ Poverty_Incidence_pct +
                                         Illiteracy_Rate_pct +
                                         NHIS_Coverage_Women_2019_pct

  cat("         Selecting GWR bandwidth (AICc)...\n")
  bw <- bw.gwr(gwr_formula, data = sp_df, approach = "AICc",
               kernel = "gaussian", adaptive = TRUE, longlat = FALSE)
  cat(sprintf("         Optimal adaptive bandwidth: %d neighbours\n", round(bw)))

  gwr_res <- gwr.basic(gwr_formula, data = sp_df, bw = bw,
                        kernel = "gaussian", adaptive = TRUE, longlat = FALSE)

  lr2 <- gwr_res$SDF$Local_R2
  cat(sprintf("         GWR mean local R-squared = %.4f (range %.3f-%.3f)\n",
              mean(lr2), min(lr2), max(lr2)))
  cat(sprintf("         Mean beta (Poverty):    %.4f\n",
              mean(gwr_res$SDF$Poverty_Incidence_pct)))
  cat(sprintf("         Mean beta (Illiteracy): %.4f\n",
              mean(gwr_res$SDF$Illiteracy_Rate_pct)))
  cat(sprintf("         Mean beta (NHIS Cov):   %.4f\n\n",
              mean(gwr_res$SDF$NHIS_Coverage_Women_2019_pct)))
}

# --- SPATIAL REGRESSION DIAGNOSTICS (SLM / SEM) ---

cat("[SPATIAL REGRESSION] OLS -> LM tests -> SLM / SEM\n")

ols_form <- Uninsurance_Rate_pct ~ Poverty_Incidence_pct +
                                    Illiteracy_Rate_pct +
                                    NHIS_Coverage_Women_2019_pct
ols_fit <- lm(ols_form, data = df_spatial)
cat(sprintf("         OLS R-squared = %.4f\n", summary(ols_fit)$r.squared))

lm_tests <- lm.LMtests(ols_fit, lw,
                        test = c("LMlag", "RLMlag", "LMerr", "RLMerr", "SARMA"))
cat("\n         Lagrange Multiplier Tests:\n")
print(summary(lm_tests))

slm_fit <- lagsarlm(ols_form, data = df_spatial, listw = lw)
cat(sprintf("\n         SLM -- rho = %.4f | AIC = %.2f\n",
            slm_fit$rho, AIC(slm_fit)))

sem_fit <- errorsarlm(ols_form, data = df_spatial, listw = lw)
cat(sprintf("         SEM -- lambda = %.4f | AIC = %.2f\n",
            sem_fit$lambda, AIC(sem_fit)))

cat(sprintf("\n         Model comparison:\n"))
cat(sprintf("           OLS: AIC = %.2f\n", AIC(ols_fit)))
cat(sprintf("           SLM: AIC = %.2f\n", AIC(slm_fit)))
cat(sprintf("           SEM: AIC = %.2f\n", AIC(sem_fit)))
best <- which.min(c(AIC(ols_fit), AIC(slm_fit), AIC(sem_fit)))
cat(sprintf("           Best fit: %s\n", c("OLS","SLM","SEM")[best]))

res_moran <- moran.test(residuals(slm_fit), lw)
cat(sprintf("\n         Residual Moran's I (SLM) = %.4f | p = %.4f\n",
            res_moran$estimate["Moran I statistic"],
            res_moran$p.value))
cat(sprintf("         %s\n",
            ifelse(res_moran$p.value > 0.05,
                   "Residual autocorrelation removed -- SLM adequate",
                   "Residual autocorrelation remains -- consider SARMA or GWR")))

cat("\n=== NHIS OOP Ghana spatial analysis complete ===\n")
