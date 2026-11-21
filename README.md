# Classifying electromyography signals with machine learning

Can four measurements from a 200 ms segment of muscle electrical activity help distinguish recordings labeled healthy, myopathy, and neuropathy? This Python project compares three supervised classifiers on that question. It covers the full workflow from reading raw electromyography (EMG) signals to interpreting where the classifiers make mistakes.

**Scope.** The experiment uses one recording per class. It measures how well a model trained on the beginning of each recording recognizes windows from the end of those same recordings. Its scores cannot establish diagnostic accuracy on new patients.

[Open the analysis notebook](./Sharif_EMG_corrected.ipynb)

## Run the notebook

Keep `README.md` and `Sharif_EMG_corrected.ipynb` in the same directory. The notebook retrieves the recordings from PhysioNet, so you need an internet connection during data loading.

**Google Colab.** Upload the notebook to Colab and select **Runtime → Run all**. The first cell installs the main dependencies into the active kernel.

**Local Jupyter environment.** From the project directory on macOS or Linux, run the following commands.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install jupyterlab numpy scipy pandas matplotlib scikit-learn wfdb
python -m jupyter lab Sharif_EMG_corrected.ipynb
```

Run the cells in order. The class counts and matrix dimensions appear first, followed by the evaluation for each model. All figures and the final comparison table appear inside the notebook. Package versions are unpinned, so exact results may vary across environments.

## Data

The notebook uses version 1.0.0 of PhysioNet’s *Examples of Electromyograms*. WFDB reads the physical signal values and the sampling frequency from each record. The notebook assigns the record’s label to every window extracted from it. [PhysioNet dataset and attribution](https://physionet.org/content/emgdb/1.0.0/)

| WFDB record | Class label | Integer used in the code |
| --- | --- | --- |
| `emg_healthy` | Healthy | 0 |
| `emg_myopathy` | Myopathy | 1 |
| `emg_neuropathy` | Neuropathy | 2 |

## From a recording to a training example

1. **Split by time.** The first 80% of the raw samples in each recording form its training portion. The last 20% form its test portion. All three classes therefore appear on both sides of the split.
2. **Filter each portion separately.** A Butterworth band-pass filter designed with order 4 retains the 20–450 Hz band. SciPy’s `filtfilt` applies it forward and backward to each channel. Filtering after the split prevents samples across the boundary from influencing one another.
3. **Create windows.** The window length is `round(0.2 * fs)` samples, where `fs` is the recording’s sampling frequency in Hz. The step is half a window, giving approximately 50% overlap. Only complete windows are kept. No window crosses the train/test boundary.
4. **Extract features.** Each complete window becomes one row in the feature matrix. The notebook calculates four values per channel and concatenates them. A window with one channel therefore produces four columns.

| Feature | Calculation within one channel and window | What it summarizes |
| --- | --- | --- |
| Mean absolute value (MAV) | Mean of the absolute sample values | Average signal magnitude, regardless of sign |
| Root mean square (RMS) | Square root of the mean squared sample value | Signal magnitude with greater weight on larger amplitudes |
| Variance | Mean squared deviation from the window mean | Spread of the signal around its local mean |
| Zero crossings | Count of adjacent samples whose product is negative | Sign changes within the window. Pairs containing an exact zero do not count |

## Models

Each model receives the same training and test windows. A scikit-learn pipeline fits `StandardScaler` on the training features and applies that transformation to the test features.

| Classifier | Configuration in the notebook | Role in the comparison |
| --- | --- | --- |
| Logistic regression | Up to 1,000 iterations | Linear baseline for the four feature types |
| Random forest | 200 trees, random seed 42 | Tests whether combinations of feature thresholds separate the classes |
| Support vector machine | RBF kernel by default, probability estimates enabled, random seed 42 | Tests a nonlinear decision boundary and supplies scores for the ROC curves |

## Reading the results

The class distribution reports **windows per class**. A longer recording can contribute more windows, so accuracy can give more weight to one recording. Read the final accuracy alongside macro F1 and the per-class results.

| Notebook output | How to interpret it |
| --- | --- |
| Classification report | Precision measures how often a predicted label is correct. Recall measures how many windows of that class were identified. Support is the number of test windows with that label. |
| Confusion matrix | Rows are the assigned recording labels. Columns are predictions. For example, the myopathy row and neuropathy column counts myopathy windows predicted as neuropathy. |
| Per-class ROC curves | Each curve treats one class as positive and the other two as negative. It shows how true- and false-positive rates change as the score threshold varies. |
| Final comparison table | Accuracy is the fraction of correctly labeled test windows. Macro F1 averages the three class-specific F1 scores equally. Macro AUC averages the three one-vs-rest ROC areas equally. |

## What the experiment can establish

Splitting before windowing keeps raw samples from appearing in both sets. However, neighboring windows within each set still overlap, and train and test windows share the characteristics of their source recording. Even a high score could reflect a recording-specific amplitude pattern. With only one recording per class, the experiment cannot distinguish that explanation from a pattern that generalizes across people.

To investigate patient-level generalization, the dataset would need multiple independent participants per class. All recordings from a given participant would then need to stay together when forming training and test sets.
