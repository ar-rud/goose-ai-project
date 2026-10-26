# 🪿Semantic segmentation of unstructured scenes, based on GOOSE dataset


## Team

| Name | GitHub | Role |
|---|---|---|
| Artur Rudish | [@ar-rud](https://github.com/ar-rud) | training, metrics evaluation, prediction |
| Denys Marchenko | [@SpyderEater](https://github.com/SpyderEater) | Data pipeline, modeling, lightweight Linknet exploring |

## Data

<img width="2339" height="693" alt="image" src="https://github.com/user-attachments/assets/0f51e4f8-41b8-4c26-b380-882e9e393b2b" />



### [GOOSE dataset](https://goose-dataset.de/) structure:

```
├── images
|   ├── test
|       ├── scenario01
|       ├── scenario02
|       ├── ...
|       └── scenario_N
|   ├── train
|   └── val
├── labels
|   ├── train
|   └── val
└──
```

images direcory contains camera photos of different scenarios.

labels directory has 3 types of masks:
- _color       -- 64-class colored segmentation mask 
- _labelids    -- 64-class grayscale mask
- _instanceids -- grayscale mask that saves only specific objects(not used for our current purposes)

### Classes structure

<img width="902" height="863" alt="image" src="https://github.com/user-attachments/assets/0942c345-c658-4dae-9a45-dd83db59f17b" />


## Baseline approach

<img width="1228" height="701" alt="image" src="https://github.com/user-attachments/assets/4a05071d-32d5-43e5-8d16-fe989de5c0fb" />


As a baseline model for semantic segmentation we used Unet, standart for such purposes due to its "skip connections" mechanism that allows decoder to get detailed data from encoder.


## Linknet experiment

Also, we explored computing capabilities of lightweight model - Linknet, also segmentation model. We used 10 times reduced dataset, ran it on 10 epochs for about 10 minutes (no GPU used). Surely, due to reduced datatset there is significant probability of missing rare categories such as "human" or "sign".

| Epoch | Train Loss | Val Loss | mIoU | Pixel Accuracy (%) |
| :---: | :---: | :---: | :---: | :---: |
| **1 / 10** | 2.4985 | 2.2624 | 0.0820 | 36.31% |
| **2 / 10** | 1.9592 | 1.5831 | 0.0879 | 56.23% |
| **3 / 10** | 1.4429 | 1.2079 | 0.1783 | 72.92% |
| **4 / 10** | 1.1966 | 1.0519 | 0.1846 | 73.94% |
| **5 / 10** | 1.0528 | 0.8923 | 0.1886 | 77.39% |
| **6 / 10** | 0.9037 | 0.8197 | 0.1897 | 77.97% |
| **7 / 10** | 0.8622 | 0.7861 | 0.1920 | 77.70% |
| **8 / 10** | 0.7759 | 0.7181 | 0.1932 | 78.29% |
| **9 / 10** | 0.7408 | 0.6822 | **0.1969** | **78.85%** |
| **10 / 10**| **0.7171** | **0.6608** | 0.1927 | 78.22% |


## AI usage

Claude - data preprocessing
Gemini - data preprocessing, also metrics evaluation discussed

**What we didn't delegate:** the model-selection decision itself (§Results) — specifically, noticing that the single-split ranking disagreed with the cross-validated ranking, and choosing to trust CV — was made by reading the actual numbers, not accepted from a tool's suggestion. Any hyperparameter or architecture choice that ended up in `config/default.yaml` was verified by rerunning `python scripts/run_experiment.py` ourselves, not taken on faith.

## Contribution statement

| Member | Contribution |
|---|---|
| Artur Rudish | 60% |
| Denys Marchenko | 40% |

Both members agree with the above distribution.


