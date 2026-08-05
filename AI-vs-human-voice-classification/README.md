# AI vs Human Voice Classification

Distinguishing synthesised speech from human speech using hand-engineered acoustic features rather than a deep audio model. Extracts pitch, spectral, and energy descriptors with librosa, compares three classifiers, and runs formal hypothesis tests on which acoustic properties actually differ between the two groups.

Dataset: 200 short utterances (100 AI, 100 human) of scripted sentences, supplied as a zip of WAV files with an accompanying `metadata.csv` mapping each file to its label and transcript.

## Feature extraction

Audio is loaded at 16 kHz mono and silence-trimmed at 25 dB. For each clip the pipeline computes:

- **Pitch**: mean, standard deviation, and 5th-to-95th-percentile range, using librosa's YIN estimator restricted to 50 to 400 Hz and masked to frames above the 35th energy percentile, so silence does not distort the estimate.
- **Spectral**: centroid, bandwidth, 85% rolloff, and flatness.
- **Energy and timing**: RMS, zero crossing rate, and clip duration.
- **Articulation rate**: characters per second and words per second, derived from the transcript length divided by duration.

## Results

Stratified 80/20 split, 40 test clips:

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Random Forest** | 0.975 | 0.952 | 1.000 | **0.976** |
| Support Vector Machine | 0.950 | 0.909 | 1.000 | 0.952 |
| Logistic Regression | 0.925 | 0.870 | 1.000 | 0.930 |

All three catch every AI clip; they differ only in how many human clips they wrongly flag as synthetic. Random Forest misses on one.

## Hypothesis tests

Welch t-tests across three hypothesis families, AI against human:

| Feature | AI mean | Human mean | p |
|---|---|---|---|
| Pitch std | 43.0 | 45.9 | 0.0075 |
| Pitch range | 126.7 | 139.6 | 0.00041 |
| Spectral centroid | 1490.8 | 1799.9 | 2.5e-16 |
| Spectral bandwidth | 1361.5 | 1493.3 | 2.5e-13 |
| Spectral rolloff | 2726.9 | 3205.4 | 1.2e-14 |
| Chars per second | 15.97 | 14.25 | 2.9e-14 |
| Words per second | 2.59 | 2.31 | 0.0026 |
| Duration | 2.64 | 2.98 | 0.00012 |

The pattern is consistent: synthetic speech is spectrally darker (lower centroid, bandwidth, and rolloff), flatter in pitch, and faster. Spectral features separate the two groups far more decisively than pitch does, which matches the model's feature importances.

## Caveats

The top-ranked feature, characters per second (importance 0.21), is computed from the transcript in the metadata rather than from the audio itself. It is a legitimate signal here because AI and human speakers read the same scripted sentences, so it measures speaking rate. But it would not be available at inference time on an unlabelled clip, so a deployable version of this classifier needs to drop it or replace it with an audio-derived rate estimate.

Beyond that: 200 clips is small, and 40 test clips means one flip moves accuracy by 2.5 points, so the ranking between the three models is not firm. All clips are short scripted sentences recorded under controlled conditions, so performance on spontaneous or noisy speech is untested, as is generalisation to TTS systems other than the one that produced this data. The eight t-tests carry no multiple-comparison correction, though most p-values survive it comfortably.

## Running it

```bash
pip install librosa numpy pandas scikit-learn scipy seaborn matplotlib
```

Upload the dataset zip and update `ZIP_PATH` and `BASE_DIR` at the top of the notebook.

## Possible extensions

Use the `split` column already present in `metadata.csv` instead of a fresh random split, so results are comparable across runs of the dataset. Add MFCCs, which are the standard baseline feature set for this task. Test against a TTS engine not represented in training to measure whether the model learned "synthetic speech" or just one vendor's artefacts.
