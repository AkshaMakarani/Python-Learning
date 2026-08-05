# EuroSAT Land Use Classification

End-to-end deep learning pipeline for multi-class land use classification on Sentinel-2 satellite imagery. Three models are built and compared — a baseline CNN, an improved CNN, and a fine-tuned ResNet50 — with augmentation, class weighting, hyperparameter search, and Grad-CAM explainability.

Built with TensorFlow / Keras on the [EuroSAT dataset](https://github.com/phelber/eurosat) (Helber et al., 2019): 27,000 labelled 64×64 RGB image patches across 10 land use classes, loaded via TensorFlow Datasets with a 70/15/15 split.

`AnnualCrop` · `Forest` · `HerbaceousVegetation` · `Highway` · `Industrial` · `Pasture` · `PermanentCrop` · `Residential` · `River` · `SeaLake`

## Approach

| Stage | What it does |
|---|---|
| Preprocessing | Two `tf.data` pipelines — 64×64 normalised for the custom CNNs, 224×224 with ResNet preprocessing for transfer learning |
| Augmentation | Flip, rotation, zoom, brightness, contrast — training split only |
| Imbalance handling | Inverse-frequency class weights, macro F1 as the primary metric |
| Model 1 | Baseline CNN — 3 conv blocks, dropout, dense head |
| Model 2 | Improved CNN — 4 blocks, BatchNorm, global average pooling, L2 regularisation |
| Model 3 | ResNet50 transfer learning — two-phase: frozen feature extraction, then fine-tuning the top 30 layers at LR 1e-5 |
| Tuning | Keras Tuner `RandomSearch` over dropout, dense units, and learning rate |
| Explainability | Grad-CAM heatmaps over the final residual block |

## Results

Across 11 configurations varying architecture, augmentation, learning rate, batch size, and regularisation:

| Model | Val Accuracy | Macro F1 |
|---|---|---|
| Baseline CNN | ~0.86 | ~0.85 |
| Improved CNN (BN + L2) | ~0.89 | ~0.88 |
| **ResNet50 (fine-tuned top-30)** | **~0.95** | **~0.95** |

Transfer learning gave the largest single gain, and fine-tuning beat frozen feature extraction — but only at a low learning rate (1e-3 during fine-tuning degraded performance by ~4 points). Augmentation helped consistently.

Grad-CAM confirms the model attends to semantically meaningful regions: road geometry for Highway, canopy texture for Forest, water bodies for River and SeaLake, rooftop grid patterns for Residential.

## Limitations

`Forest` vs `HerbaceousVegetation` and `PermanentCrop` vs `AnnualCrop` remain the dominant confusions — they share visual signatures at 64×64. Upscaling to 224×224 introduces interpolation artefacts rather than real detail. The dataset covers European land under limited seasonal conditions, so performance on other geographies is untested. Grad-CAM shows plausible attention, not formal attribution.

## Running it

Open in Google Colab with a GPU runtime (Runtime → Change runtime type → GPU). TensorFlow, `tensorflow-datasets`, and `keras-tuner` all ship with Colab — the notebook deliberately avoids `pip install`, which can break the protobuf version and crash the tfds import. The dataset downloads automatically via TFDS; no manual download needed.

Locally you'll need `tensorflow`, `tensorflow-datasets`, `keras-tuner`, `scikit-learn`, `pandas`, `matplotlib`, and a GPU for the ResNet50 stage.

## References

- Helber, P., Bischke, B., Dengel, A., & Borth, D. (2019). *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification.* IEEE JSTARS.
- He, K. et al. (2016). *Deep Residual Learning for Image Recognition.* CVPR.
- Selvaraju, R. R. et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV.
