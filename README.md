# Batch Size and Calling Interface in Gradient Boosted Tree Serving

Replication package for the paper *The Effect of Batch Size and Calling Interface on
Gradient Boosted Tree Serving Cost*.

The study measures per-row inference latency for LightGBM, XGBoost and CatBoost across
nine serving interfaces and thirteen batch sizes, at three model sizes, on a single
thread. Every interface holds the same trained model, so latency differences are
attributable to the serving path rather than to the model. The full grid was run twice on
separately allocated cloud instances.

Everything reported in the paper can be regenerated from this repository, either by
re-running the experiment or by re-analysing the measurements that are included here.

## Quick start

Reproduce every table and reported number from the included measurements. Takes seconds
and needs only pandas and numpy.

```bash
pip install -r requirements.txt
python scripts/analyze_results.py
```

Redraw all eight figures from the same measurements.

```bash
python scripts/make_figures.py
```

Re-run the experiment from scratch. Open `notebooks/02_full_experiment.ipynb` in Google
Colab, confirm the runtime is CPU rather than GPU, and run all cells. Around forty to
sixty minutes on a two-core instance.

## Layout

```
notebooks/
  01_smoke_test.ipynb         short rehearsal, about five minutes, checks the harness
  02_full_experiment.ipynb    the full grid, writes results and figures to Drive
scripts/
  analyze_results.py          regenerates every table and number in the paper
  make_figures.py             regenerates all eight figures
results/
  primary/                    the run the paper reports
  replication/                the independent replication
figures/                      output of make_figures.py
docs/
  EXPERIMENT_NOTES.md         measurement protocol and design decisions
```

## What was measured

Three libraries, nine serving interfaces:

| Library | Interfaces |
|---|---|
| LightGBM | sklearn wrapper, native `Booster.predict`, ONNX Runtime |
| XGBoost | sklearn wrapper, `DMatrix` predict, `inplace_predict`, ONNX Runtime |
| CatBoost | sklearn wrapper, ONNX Runtime |

Crossed with thirteen batch sizes from 1 to 16,384, three tree counts (100, 500, 1000),
two feature counts on synthetic data (20 and 100), and one and two threads. Six
configurations, 702 measurements per run.

The prediction task is Online Shoppers Purchasing Intention from the UCI repository,
12,330 browsing sessions with seventeen features, predicting whether a session ended in a
purchase. Synthetic data is used only for the feature-count comparison, since the column
count of a real dataset cannot be varied.

## Re-running the experiment

The notebook writes to a folder named by the `RUN_ID` string in the configuration cell.
Resume works by reading back whatever is already saved under the current run id, so
re-running with an unchanged id after a complete run measures nothing and leaves the drift
log empty. Change the string to start a clean run. The notebook prints how many
measurements it is about to take before the loop starts, and warns when that number is
zero.

Results are written to local disk first and mirrored to Google Drive after each
configuration, so an unmounted Drive interrupts the mirroring rather than the experiment.

## Measurement protocol

The design decisions that shape the numbers are described in `docs/EXPERIMENT_NOTES.md`.
In brief: thread limits are set before any library import, configurations run in shuffled
order, twenty warm-up calls are discarded, each measurement takes eleven timed repeats with
every call timed individually, a calibration step equalises wall time across batch sizes,
one persistent reference model is re-measured throughout the session to record host drift,
and every interface is checked against its library's own call before any timing is
recorded.

## Data files

`results/primary/` holds the run the paper reports.

| File | Contents |
|---|---|
| `results.csv` | 702 measurements, one row per configuration, path and batch size |
| `correctness.csv` | 36 agreement comparisons between serving paths |
| `noise_reference.csv` | 37 measurements of the persistent drift reference |
| `crossover_robustness.csv` | winning margin against measurement noise for every comparison |
| `zipmap_ablation.csv` | CatBoost exported-model output-node ablation |
| `environment.json` | processor and library versions |

`results/replication/results.csv` holds the independent replication.

Column meanings for `results.csv`:

| Column | Meaning |
|---|---|
| `config` | configuration identifier, dataset with feature count, tree count and thread count |
| `sweep` | which experimental factor this configuration varies |
| `path` | serving interface |
| `batch` | rows per prediction call |
| `n_calls` | calls per repeat, chosen by calibration |
| `median_us_per_row` | median per-row latency across eleven repeats, microseconds |
| `iqr_us_per_row` | interquartile range across repeats, the noise estimate |
| `min_us_per_row`, `max_us_per_row` | extremes across repeats |
| `call_p50_us`, `call_p95_us`, `call_p99_us` | percentiles of individual call latency |

## Environment

Measurements come from Google Colab, free tier: Intel Xeon at 2.20 GHz, 2 cores, Python
3.13.15, LightGBM 4.7.0, XGBoost 3.4.1, CatBoost 1.2.10, ONNX Runtime 1.24.4. Absolute
microsecond values belong to that hardware. The comparative results are what the paper
claims.

## Maintainer

This repository is maintained by Muhammad Ahmad Zia, corresponding author for the paper. He
is a lecturer in the Department of Computer Science & IT at the University of Lahore. His
work covers green machine learning, energy-efficient and sustainable model serving,
mixed-precision inference, gradient boosted tree benchmarking, and recommendation systems
for e-commerce.

Questions about the measurements, requests for the raw session logs, or reports of a
reproduction failure are all welcome. Open an issue on this repository, or write to
hello@ahmadzia.com.

- Website: [www.ahmadzia.com](https://www.ahmadzia.com)
- ORCID: [0000-0002-3208-6246](https://orcid.org/0000-0002-3208-6246)

## Authors

Marwan Abu-Zanona, Maryam Al-Dairi, Lamia Hassan Rahamatalla, Najla Abdulaziz Almousa and
Hebah Abdullah Abubakr, Department of Management Information Systems, College of Business
Administration, King Faisal University, Al-Ahsa, 31982, Saudi Arabia.

Muhammad Ahmad Zia, corresponding author, Department of Computer Science & IT, The
University of Lahore, Lahore, 54000, Pakistan.

Atif Ikram, Department of Computer Science & IT, The University of Lahore, Lahore, 54000,
Pakistan, and Faculty of Computer Science and Mathematics, Universiti Malaysia Terengganu,
Malaysia.

## Citation

See `CITATION.cff`. Please cite both the paper and this archive.

## License

MIT, see `LICENSE`. The Online Shoppers Purchasing Intention dataset is distributed by the
UCI Machine Learning Repository under its own terms and is downloaded by the notebook
rather than redistributed here.
