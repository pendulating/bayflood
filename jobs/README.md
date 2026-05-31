# Cluster job scripts (`jobs/`)

These are **example SLURM submission scripts** for the authors' compute cluster.
They are provided for transparency and as a starting point — they are **not
portable** and must be edited for your environment before use.

Machine-specific values have been replaced with placeholder shell variables so
the scripts carry no author- or cluster-specific paths. Set these to match your
environment before submitting:

| Placeholder | Meaning | Example |
|---|---|---|
| `$REPO_ROOT` | This repository's clone path | `/home/you/bayflood` |
| `$HOME` | Your home directory (used for `.bashrc`) | provided by the shell |
| `$VLM_REPO_ROOT` | Clone of the VLM-inference code (cambrian/janus scripts) | `/home/you/street-flooding` |
| `$CONDA_ENV_ROOT` | Directory holding the conda envs | `/opt/conda/envs` |
| `${USER}@example.com` | `--mail-user` notification address | your email |

```bash
export REPO_ROOT=/path/to/your/bayflood
sbatch jobs/fit.sub my_run
```

> **Note:** `#SBATCH` directive lines (e.g. `-o`/`-e` log paths) are parsed by
> Slurm *before* the shell runs and therefore **do not expand shell variables**.
> Edit those lines to a literal path for your cluster. The `--partition` value
> is also cluster-specific. These scripts are provided for transparency; the
> actual work each runs is the standard CLI documented in
> `docs/CLI_REFERENCE.md` and the project `README.md`:

| Script | Runs |
|---|---|
| `fit.sub`, `fit_cov_geo.sub` | ICAR model fit (with covariates) |
| `fit_no_covariates.sub`, `fit_nocov_geo.sub` | ICAR model fit (no covariates) |
| `downsampled.sub`, `downsampled_all.sub` | Downsampling robustness runs |
| `baselines.sub`, `baselines_sequence.sh` | Post-processing baseline comparisons |
| `trimmed.sub` | Trimmed/auxiliary run |
| `labelling-server.sub` | Label Studio annotation server |
