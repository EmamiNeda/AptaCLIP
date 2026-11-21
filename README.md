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


## Setup

Use a Linux GPU environment (such as Colab), with `cd-hit` installed separately. Install Python packages with:

```bash
pip install -r requirements.txt
```

Provide **the exact original** `data/train_set.csv` and `data/test_set.csv` with `Target_Sequence`, `Aptamer_Sequence`, and `Class` columns.The notebook rebuilds a CD-HIT-based split from these files; it does not simply use the supplied split as the independent test.

## Run

From the repository root:

```bash
python src/train.py
```


## Architecture

- Frozen DNABERT-2-117M: BPE nucleotide tokenization, RNA U→T conversion, masked mean pooling to 768 dimensions.
- Frozen ESM2-650M: chunked protein encoding (up to 1022 residues/chunk), length-weighted aggregation, 1280 dimensions.
- Modality-specific MLPs: 512 → 256 → 128, batch normalization, dropout and L2 normalization.
- Classification head: concatenated protein projection, classification aptamer projection, RNA/DNA indicator → Dense(128) → sigmoid.
- Protein-anchored triplet loss: squared Euclidean distance between normalized projections, with margin selected on validation.
- Classification BCE and triplet loss weighted by a validation-selected configuration.


