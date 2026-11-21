# EMG signal classification

This project compares logistic regression, a random forest, and a support vector machine on three electromyography (EMG) recordings. Each model uses four time-domain features to classify 200 ms windows from recordings labeled healthy, myopathy, and neuropathy.

[Analysis notebook](./Bita_HERAVI_ML_Project.ipynb)

## Dataset

The data come from PhysioNet’s *Examples of Electromyograms*, version 1.0.0. The notebook reads the signals and their sampling frequencies directly with WFDB. Each window inherits the label of its source recording. [Dataset on PhysioNet](https://physionet.org/content/emgdb/1.0.0/)

| Recording | Label |
| --- | --- |
| `emg_healthy` | Healthy |
| `emg_myopathy` | Myopathy |
| `emg_neuropathy` | Neuropathy |

## Signal processing

The first 80% of each recording supplies the training data and the final 20% supplies the test data. Each portion is filtered independently with a 20–450 Hz Butterworth band-pass filter applied forward and backward. This prevents samples on one side of the split from affecting filtered values on the other.

Each filtered signal is divided into 200 ms windows with approximately 50% overlap. Window length in samples is calculated from the recording’s sampling frequency. Only complete windows are retained. No window spans the training and test portions.

## Features

Each row of the feature matrix represents one window. The four features are calculated separately for each signal channel.

| Feature | Definition |
| --- | --- |
| Mean absolute value | Average absolute amplitude over the window |
| Root mean square | Square root of the mean squared amplitude, giving larger amplitudes greater weight |
| Variance | Mean squared deviation from the window’s average amplitude |
| Zero crossings | Number of consecutive sample pairs with opposite signs |

## Model comparison

All three classifiers use the same chronological split. Each scikit-learn pipeline fits its feature scaler on the training windows.

| Classifier | Configuration |
| --- | --- |
| Logistic regression | Linear baseline with a maximum of 1,000 iterations |
| Random forest | 200 decision trees, random seed 42 |
| Support vector machine | Radial basis function kernel, probability estimates enabled, random seed 42 |

## Evaluation

For each classifier, the notebook produces a per-class precision, recall, and F1 report, a confusion matrix, and one-vs-rest ROC curves. The final table compares accuracy, macro F1, and macro ROC AUC. Accuracy weights each test window equally. The macro scores give each of the three classes equal weight, even when the recordings yield different numbers of windows.

The unit of evaluation is a signal window. Because each class comes from a single recording, class identity and recording identity coincide. A model could distinguish the recordings using recording-specific amplitude differences. The experiment therefore measures classification on later segments of the same recordings. Assessing generalization to new patients would require multiple participants per class and a participant-level split.

## Reproducibility

The notebook uses Python, WFDB, NumPy, SciPy, pandas, Matplotlib, and scikit-learn. Data loading requires internet access. The following commands start a local JupyterLab environment on macOS or Linux.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install jupyterlab wfdb numpy scipy pandas matplotlib scikit-learn
python -m jupyter lab Sharif_EMG_corrected.ipynb
```
