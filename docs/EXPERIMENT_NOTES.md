# Measurement Protocol and Design Decisions

This document records why the experiment is built the way it is. Several choices exist to
defend the measurements against known failure modes in performance benchmarking, and they
are easier to evaluate when the reasoning is written down next to the code.

## The comparison being made

One trained model per library per tree count feeds every serving interface for that
library. A difference in latency between two interfaces therefore cannot come from a
difference in the model. This is what makes the study a comparison of serving paths rather
than a comparison of libraries.

The exported ONNX paths change more than the calling convention. They also change the
execution runtime and the model representation. The paper attributes differences to the
serving path as a whole rather than to any single component of it.

## Thread pinning

`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` and `NUMEXPR_NUM_THREADS` are
set before any library is imported. Setting them afterwards has no effect, since the thread
pools are created at import time. Every library and every ONNX Runtime session is confined
to one thread for the main grid.

Single-thread pinning is what makes interface attribution possible. With default all-core
threading, a difference between two paths could come from how well each one parallelises
rather than from what each one does per call.

## Calibration

A fixed row count per measurement would make batch size one dominate the run, since it
needs thousands of calls to move the same number of rows. The harness times a few calls
first, then picks a call count that fills a budget of roughly 0.15 seconds. A batch of one
and a batch of 16,384 then cost about the same in wall time.

Call counts are clamped between 5 and 3000 to keep very fast and very slow paths from
producing degenerate measurements.

## Warm-up and repeats

Twenty calls are executed and discarded before timing starts, because the first call
through any path pays one-off allocation and lazy initialisation costs.

Discarding a warm-up does not guarantee a steady state. Managed runtimes often fail to
reach one at all, so this study reports the interquartile range of every measurement rather
than assuming stability after warm-up.

Eleven timed repeats follow. Each call inside a repeat is timed on its own rather than
timing the pass as a whole, which costs almost nothing and yields the tail of the call
distribution as well as the middle.

## Shuffled configuration order

A shared cloud host can slow down partway through a session. Running the grid in its
natural order would let that slowdown land disproportionately on whichever interface
happened to be scheduled late. Configurations therefore run in a shuffled order with a
fixed seed, which spreads any drift across all paths.

## Persistent drift reference

One reference model is trained before the loop starts and held for the whole session. It is
re-measured at a fixed batch size roughly every twenty measurements, plus once at session
start and once at session end.

The reference must persist. An earlier version of this experiment re-trained the reference
model for each configuration, which meant the drift log tracked variation within a
configuration but could not be compared across them. The current design fixes the reference
so that a single time series covers the whole session.

## Correctness gate

Before any timing is recorded, every interface is checked against its own library's wrapper
call on the same rows, with agreement required to within 1e-4. A faster path that returns
different numbers is a bug rather than a speedup.

Results in `correctness.csv` show 36 comparisons per run with zero disagreements and a
largest deviation of 4.8e-7.

## Robustness criterion

At any batch size the cheapest interface can be identified, but near a crossover two
interfaces cost almost the same and the difference sits inside the noise. A winner is
counted as robust only when its margin over the runner-up exceeds three times the
interquartile range of its own measurement. Comparisons that fail are reported as having no
winner rather than being assigned one.

This threshold was chosen after inspecting the measurement spread, not before data
collection. It is a reporting rule applied uniformly to every comparison, not a
pre-registered test.

## The CatBoost ablation

CatBoost's exported model ends in a node that builds a dictionary object for each row of
output, which could inflate its exported-runtime cost without any of it being inference.
The node is removed by editing the computation graph and the model is re-timed. Ratios
between the two versions sit near one at every batch size, so the cost remains after the
node is removed.

This ablation runs inside the experiment notebook rather than separately, so the result is
reproducible from this repository.

## Two runs

The full grid was executed twice on separately allocated cloud instances. Absolute
latencies differ between them because the instances differ in speed, while the ordering of
the nine paths is identical and most ratios are close. Where the two runs disagree, the
paper reports both.

## Known limitations

Measurements come from one machine, a two-core shared cloud instance, so absolute
microsecond values will not transfer.

The attribution of cost to the serving path is an inference from how latency responds to
model size, not a direct measurement of what each layer spends its time on. Profiling the
wrappers would settle it.

The feature-count comparison uses synthetic data and replicates for only one of the three
libraries. It is reported as secondary.

Energy was not measured, since the hardware counters are not exposed inside the container.
