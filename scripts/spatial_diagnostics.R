# scripts/spatial_diagnostics.R — NHIS OOP Ghana 261 Districts | AIPOCH v6.5
# Standalone R spatial diagnostics script.
# Delegates to ../analysis.R from project root context.
# Usage: Rscript scripts/spatial_diagnostics.R

# Navigate to project root if called from scripts/
if (basename(getwd()) == "scripts") {
  setwd("..")
}
source("analysis.R")
