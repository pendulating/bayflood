from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent

CURRENT_DF = str(BASE_DIR / "runs/icar_icar/simulated_False/ahl_True/covariates_True/MAY30_COV_20260530-2312/analysis_df_MAY30_COV_05312026.csv")

CURRENT_NO_COVARIATES_DF = str(BASE_DIR / "runs/icar_icar/simulated_False/ahl_True/covariates_False/MAY30_NOCOV_20260530-2309/analysis_df_MAY30_NOCOV_05302026.csv")

PAPER_PATH = str(BASE_DIR / 'papers/natcities_bayflood_2025')

DELIVERABLES_PATH = str(BASE_DIR / 'deliverables')
WGS = 'EPSG:4326'
PROJ = 'EPSG:2263'
GEO_PATH = '../../aggregation/geo/data'

# Post-adjacency-fix baseline batch (FINAL_COMMS_BASELINES_MAY31_*). Only
# converged runs write performance_on_baselines.csv; the notebook further pins
# to the 20 most recent. The older FINAL_COMMS_BASELINES_2026013*/02* runs
# predate the adjacency fix and are intentionally excluded.
CURRENT_PP_BASELINES_GLOB = str(BASE_DIR / "runs/icar_icar/simulated_False/ahl_True/covariates_True/FINAL_COMMS_BASELINES_MAY31_*/performance_on_baselines.csv")

CURRENT_ADJ_1 = str(BASE_DIR / "data/adjacency/cg_500/ct_nyc_adj_list_custom_geometric_node1.txt")
CURRENT_ADJ_2 = str(BASE_DIR / "data/adjacency/cg_500/ct_nyc_adj_list_custom_geometric_node2.txt")

ESTIMATE_TO_USE = 'confirmed_or_above_thres'
