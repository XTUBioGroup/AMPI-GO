# AMPI-GO

**AMPI-GO: Interpretable Protein Function Prediction through Attention-Driven Multi-Modal Fusion from Residue to Network Levels**

AMPI-GO is a deep learning framework for predicting **Gene Ontology (GO) function annotations** of proteins. It fuses four complementary data modalities — protein sequence, structure, InterPro domain features, and protein-protein interaction (PPI) networks — via a multi-modal attention-based architecture.

---

## Table of Contents

- [Dataset Sources](#dataset-sources)
- [Project Structure](#project-structure)
- [Python Files Description](#python-files-description)
  - [Core Modules](#core-modules)
  - [Data Preprocessing Scripts](#data-preprocessing-scripts)
- [Installation](#installation)
- [How to Run](#how-to-run)
  - [Data Preparation Pipeline](#data-preparation-pipeline)
  - [Training](#training)
- [Dependencies](#dependencies)

---

## Dataset Sources

### 1. PDB Dataset (from GAT-GO)

The first dataset originates from **GAT-GO** and consists of experimentally determined protein structures from the **Protein Data Bank (PDB)**. Protein sequences and their corresponding PDB chain identifiers are used to derive residue-level structural features (Cα distance matrices).

### 2. AFDB Dataset (from DPFunc)

The second dataset originates from **DPFunc** and leverages predicted structures from the **AlphaFold Protein Structure Database (AFDB)**. This dataset expands coverage to proteins without experimentally solved structures.

### Data Modalities & External Databases

| Modality | Source | Description |
|---|---|---|
| **Protein Structures** | [PDB](https://www.rcsb.org/) & [AFDB](https://alphafold.ebi.ac.uk/) | 3D structures used to construct residue-level Cα distance graphs |
| **Protein-Protein Interactions** | [STRING Database](https://string-db.org/) | PPI networks with interaction scores used as edge weights |
| **Domain Annotations** | [InterProScan 6](https://www.ebi.ac.uk/interpro/) | Protein domain/motif/family annotations obtained by scanning protein sequences with InterProScan 6 |
| **Gene Ontology** | [Gene Ontology](http://geneontology.org/) | GO term annotations (MF, BP, CC) used as training labels |
| **Sequence Embeddings** | [ProtT5](https://github.com/agemagician/ProtTrans) | Pre-trained protein language model embeddings (1024-dim) |

---

## Project Structure

```
AMPI-GO/
├── AMPI_main.py                 # Main training entry point
├── GCN_model.py                 # Neural network model definitions
├── Evaluation.py                # Evaluation metrics (Fmax, AUPR) & GO ontology
├── model_utils.py               # Training utilities (FocalLoss, test loop)
├── objective.py                 # AverageMeter utility
├── environment.yml              # Conda environment specification
├── data/                        # Data directory
│   ├── go.obo                   # Gene Ontology OBO file
│   ├── all_ids_seq_final_big.txt
│   ├── pdb2go_propagate_pdb.json
│   ├── data_spilt_big/          # Train/valid/test ID splits
│   ├── data_split_pdb/          # Alternative PDB-based splits
│   ├── interpro_features_big/   # InterPro sparse feature matrices
│   └── interpro_pdb/            # PDB-based InterPro features
└── process_data/                # Data preprocessing scripts
    ├── build_PDB_Uniport.py
    ├── build_pdb_uniport_string.py
    ├── build_residue_graph.py
    ├── bulid_PPI.py
    ├── dataproicess-txt.py
    ├── download_ppi.py
    ├── pre-modle.py
    ├── process_GO.py
    ├── process_PPI.py
    ├── process_data_id.py
    ├── process_interpro.py
    ├── process_struct.py
    └── spilt_interpro.py
```

---

## Python Files Description

### Core Modules

| File | Description |
|---|---|
| `AMPI_main.py` | Main training entry point. Loads preprocessed data, trains one model per GO namespace (MF/BP/CC), and saves the best checkpoint by validation Fmax. |
| `GCN_model.py` | Neural network model definitions including GCN, GATv2, Bi-Directional Transformer, gated fusion, and the top-level multi-modal model. |
| `Evaluation.py` | Evaluation metrics (Fmax, AUPR) and GO ontology hierarchy parsing from `go.obo`. |
| `model_utils.py` | Training utilities: FocalLoss for class-imbalanced multi-label classification, test loop, and result merging. |
| `objective.py` | Lightweight `AverageMeter` class for tracking running averages during training. |

---

### Data Preprocessing Scripts

All scripts are located in the `process_data/` directory. They are intended to be run **sequentially** to build the complete data pipeline.

| Script | Purpose |
|---|---|
| `dataproicess-txt.py` | Extracts protein sequences from TFRecord files (one-hot encoded) and converts them to plain text (ID + amino acid sequence) |
| `build_PDB_Uniport.py` | Maps PDB chain IDs to UniProt accessions using SIFTS cross-reference data |
| `build_pdb_uniport_string.py` | Extends PDB→UniProt mapping to STRING database IDs via STRING's protein aliases file |
| `download_ppi.py` | Downloads and processes species-level PPI networks from STRING; builds 2-hop PPI subgraphs with a mixed Top-K strategy (Top-100 for 1-hop neighbors, Top-10 for 2-hop) |
| `bulid_PPI.py` | Constructs DGL-format PPI graphs with ProtT5 node features, STRING edge weights, and center-node marking |
| `pre-modle.py` | Generates ProtT5 (ProtTrans) protein-level embeddings with sliding-window handling for long sequences, binary search for optimal chunk size, and checkpoint resume |
| `process_GO.py` | Propagates GO annotations through the ontology hierarchy and filters annotations to the train/validation protein set |
| `process_PPI.py` | Extracts supplementary protein sequences from the STRING FASTA database for PPI neighbor nodes |
| `process_data_id.py` | Splits proteins into ontology-specific (MF/BP/CC) train/validation/test sets and generates FASTA files |
| `process_interpro.py` | Builds CSR sparse feature matrices from per-protein InterProScan TSV files; constructs domain vocabulary from the training set only |
| `spilt_interpro.py` | Creates empty placeholder TSV files for proteins without InterProScan hits, ensuring every protein has a corresponding feature file |
| `process_struct.py` | Extracts Cα distance matrices from TFRecord structure files and matches them to ProtT5 embeddings |
| `build_residue_graph.py` | Builds residue-level DGL graphs from Cα distance matrices and ProtT5 residue embeddings |

---

## Installation

### Prerequisites

- **Conda** (Miniconda or Anaconda)
- **CUDA 12.1** capable GPU (recommended for training)
- At least 32 GB of system RAM for data preprocessing

### Setup

```bash
# 1. Clone the repository
git clone <repository-url>
cd AMPI-GO

# 2. Create the conda environment
conda env create -f environment.yml

# 3. Activate the environment
conda activate protfunc
```

The environment includes:
- **Python** 3.10
- **PyTorch** 2.1.0 with CUDA 12.1
- **DGL** 2.4.0 (Deep Graph Library, GPU-enabled)
- **Transformers** 4.39.3 (HuggingFace, for ProtT5)
- **scikit-learn**, **pandas**, **numpy**, **scipy**, **networkx**, **biopython**

---

## How to Run

### Data Preparation Pipeline

The preprocessing scripts in `process_data/` must be executed in order. Adjust file paths inside each script to match your local data layout before running.

**Step 1 — Generate ProtT5 embeddings from sequences:**
```bash
python process_data/pre-modle.py
```

**Step 2 — Extract structure features (Cα distance matrices):**
```bash
python process_data/process_struct.py
```

**Step 3 — Build residue-level graphs from structure features and embeddings:**
```bash
python process_data/build_residue_graph.py
```

**Step 4 — Process InterPro domain features (pre-processed by InterProScan 6):**
```bash
python process_data/spilt_interpro.py
python process_data/process_interpro.py
```

**Step 5 — Download PPI information from STRING:**
```bash
python process_data/download_ppi.py
```

**Step 6 — Generate protein-level ProtT5 embeddings for interacting proteins in the PPI network:**
```bash
python process_data/process_PPI.py
```

**Step 7 — Build PPI graphs:**
```bash
python process_data/bulid_PPI.py
```

### Training

Once all data is prepared, launch training with:

```bash
python AMPI_main.py
```

The script will sequentially train one model for each GO namespace:

1. **MF** (Molecular Function)
2. **BP** (Biological Process)
3. **CC** (Cellular Component)

Training outputs include:
- Best model checkpoints (selected by validation Fmax)
- Training/validation loss curves
- Evaluation metrics: **Fmax** and **AUPR**

## Dependencies

See `environment.yml` for the complete, version-pinned environment specification.

Key packages:

- `pytorch=2.1.0` (CUDA 12.1)
- `dgl=2.4.0+cu121`
- `transformers=4.39.3`
- `scikit-learn=1.7.2`
- `pandas=2.3.3`
- `numpy=1.26.4`
- `scipy=1.15.3`
- `networkx=3.4.2`
- `biopython=1.87`
- `tqdm=4.67.3`

---

