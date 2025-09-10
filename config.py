import os


def _as_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


# Core dataset and adjacency defaults. Override via environment variables.
DATASET_PATH: str = os.getenv(
    "EMPIRICAL_DATA_PATH",
    "/share/ju/matt/street-flooding/aggregation/context_df_02102025.csv",
)

ADJ_NODE1_PATH: str = os.getenv(
    "ADJ_NODE1_PATH",
    "data/adjacency/cg_500/ct_nyc_adj_list_custom_geometric_node1.txt",
)
ADJ_NODE2_PATH: str = os.getenv(
    "ADJ_NODE2_PATH",
    "data/adjacency/cg_500/ct_nyc_adj_list_custom_geometric_node2.txt",
)
ADJ_NPY_PATH: str | None = os.getenv("ADJ_NPY_PATH")

EXTERNAL_COVARIATES: bool = _as_bool(os.getenv("EXTERNAL_COVARIATES"), default=False)

# Optional: default sampling settings (can still be overridden at CLI)
DEFAULT_WARMUP: int = int(os.getenv("DEFAULT_WARMUP", "1000"))
DEFAULT_SAMPLES: int = int(os.getenv("DEFAULT_SAMPLES", "1500"))


