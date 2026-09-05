# Measurements

`primary/` holds the run reported in the paper. `replication/` holds the independent
replication executed on a separately allocated cloud instance.

Both runs cover the same grid: six configurations, nine serving paths, thirteen batch
sizes, 702 measurements each. Neither has gaps.

Run `python scripts/analyze_results.py` from the repository root to regenerate every table
and reported number from these files. Column meanings are documented in the top-level
README.
