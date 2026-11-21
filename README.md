# AptaCLIP

Source-aligned code release for the revised **AptaCLIP** manuscript: sequence-based prediction of aptamer–protein interactions using **DNABERT-2-117M** (`zhihan1996/DNABERT-2-117M`) and **ESM2-650M** (`facebook/esm2_t33_650M_UR50D`).

## Structure

```text
AptaCLIP/
├── data/
├── src/
│   ├── train.py
│   ├── model.py
│   └── dataset.py
├── requirements.txt
└── README.md
```

## Provenance and scope

The training script is adapted from **code cell 2** of the supplied `4_DNA_RNA_Strafication (2)(7).ipynb`, preserving its CD-HIT 50%-identity protein clustering, length-matched putative-negative sampling, 5-seed ensemble, validation-only hyperparameter/threshold selection, and held-out test evaluation. The model and sequence cleaning functions were extracted without changing their mathematical definitions. The full notebook also contains additional experiments and analyses **not** packaged in these five files.

**Important:** The notebook's printed reference result is accuracy **0.7080**, ROC-AUC **0.7883**, AUPR **0.7520**, while the revised manuscript reports accuracy **0.7141**, ROC-AUC **0.7896**, AUPR **0.7340** for its primary independent test. These are not identical. This release should not be described as verified reproduction of every reported figure or metric until the original input CSVs, preprocessing, model selection and run outputs have been checked. No GPU end-to-end training was performed while preparing this repository.

## Setup

Use a Linux GPU environment (such as Colab), with `cd-hit` installed separately. Install Python packages with:

```bash
pip install -r requirements.txt
```

Provide **the exact original** `data/train_set.csv` and `data/test_set.csv` with `Target_Sequence`, `Aptamer_Sequence`, and `Class` columns. Do not upload confidential or unpublished data without permission. The notebook rebuilds a CD-HIT-based split from these files; it does not simply use the supplied split as the independent test.

## Run

From the repository root:

```bash
python src/train.py
```

Optionally set `APTACLIP_TRAIN`, `APTACLIP_TEST`, and `APTACLIP_OUTPUT` environment variables to change paths. The script executes a computationally expensive 7-configuration × 5-seed search, downloads two large pretrained encoders, and writes intermediate embeddings, weights, selection summaries and test predictions to the output directory. It may require more than a single small GPU.

## Architecture

- Frozen DNABERT-2-117M: BPE nucleotide tokenization, RNA U→T conversion, masked mean pooling to 768 dimensions.
- Frozen ESM2-650M: chunked protein encoding (up to 1022 residues/chunk), length-weighted aggregation, 1280 dimensions.
- Modality-specific MLPs: 512 → 256 → 128, batch normalization, dropout and L2 normalization.
- Classification head: concatenated protein projection, classification aptamer projection, RNA/DNA indicator → Dense(128) → sigmoid.
- Protein-anchored triplet loss: squared Euclidean distance between normalized projections, with margin selected on validation.
- Classification BCE and triplet loss weighted by a validation-selected configuration.


