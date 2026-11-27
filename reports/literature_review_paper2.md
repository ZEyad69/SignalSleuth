## Paper 2: Current Time Series Anomaly Detection Benchmarks are Flawed and are Creating the Illusion of Progress
Wu and Keogh. IEEE TKDE, 2021. https://arxiv.org/abs/2009.13807

### Problem
Most anomaly detection papers are tested on a few public benchmarks (Yahoo, Numenta, NASA, OMNI). The authors claim that most series in these benchmarks have at least one of four flaws, so published comparisons, and the apparent progress of deep learning, may be unreliable.

### The four flaws
1. **Triviality.** Many problems can be solved with one line of basic code.
2. **Unrealistic anomaly density.** Some series are half anomalies, some have many anomalies packed together, some have anomalies a single point apart.
3. **Mislabeled ground truth.** Examples of identical events where only one is labeled, and of labels that look arbitrary.
4. **Run-to-failure bias.** Many anomalies sit near the end of the series, so a detector that always points at the last point has a good chance of being right.

### What "trivial" means here
A problem is trivial if it can be solved with one line of standard MATLAB, using only basic operations (mean, max, std, diff, moving mean and moving std), not high-level functions like kmeans. For the whole Yahoo benchmark they find rules such as `abs(diff(TS)) > b` (with or without a moving mean and standard deviation) and report that 316 of 367 series (86.1%) are solved this way. The parameters were searched by brute force for each series. They admit the definition is informal and needs some human work.

### What they propose
- The UCR Time Series Anomaly Archive: each series has exactly one anomaly, a training part given in the file name, and the anomaly location in the name.
- Evaluation is then binary: the algorithm returns the most likely location, with some tolerance ("slop") so it is not punished for placing the score at the start or end of a window.
- Plot the data and the scores, describe the invariances an algorithm needs, and give simple older methods (for example discords) a fair chance against deep learning.

### Limits (my reading)
- The one-liner parameters are found with the labels, so they show that a problem is easy, not that a label-free detector would work.
- The paper introduces the archive; it does not run a full comparison of detectors on it. It says a small fraction of the archive may be solvable by one line, without a number.
- It does not discuss Point Adjustment; that is the first paper's topic.

### What I take for SignalSleuth
- Use UCR, with one anomaly per series and a tolerance, and report a threshold-free top-1 metric next to the threshold-based ones.
- Always compare against a trivial baseline.
- Plot the series and the scores, not only tables.
- My shape detector (distance to the nearest normal window) is close to the discord idea they mention, but it uses the training part as reference instead of the series itself.

### Is my `diff` baseline of the same kind?
Yes, it is the core of their one-liner `abs(diff(TS)) > b`. The difference: they search `b` for each series with the labels, I take the threshold from the calibration part. So it is a one-liner-style baseline, not their exact procedure.
