# Graph Contrastive Learning for Extremist Narrative Analysis

A framework for analyzing extremist social media content using graph neural networks with contrastive learning. Built upon the **SUBLIME** graph structure learning framework with context-aware extensions.

---

## 📋 Table of Contents

- [🚀 Installation](#-installation)
- [⚡ Quick Start](#-quick-start)
- [📊 Datasets](#-datasets)
- [🧠 Methodology](#-methodology)
- [📖 Usage Guide](#-usage-guide)
  - [Running Experiments](#running-experiments)
  - [Adding Context-Based Edges](#adding-context-based-edges)
  - [Sensitivity Analysis with Sweeps](#sensitivity-analysis-with-sweeps)
  - [Node Classification](#node-classification)
  - [BERT Baseline Text Classification](#bert-baseline-text-classification)
- [⚙️ Configuration](#-configuration)
- [🔧 Troubleshooting](#-troubleshooting)

---

## 🚀 Installation

### Step 1: Clone and Navigate
```bash
git clone https://github.com/diniaouri/GR-EN-A-DE.git
cd GR-EN-A-DE/Contrastive\ Learning
```

### Step 2: Set Up Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

> **⚠️ Note:** The DGL (Deep Graph Library) installation is configured for CPU. For GPU support, please refer to [DGL installation guide](https://www.dgl.ai/pages/start.html).

---

## ⚡ Quick Start

1. **Update Configuration**  
   Edit `experiment_params.csv` to set your experiment parameters.

2. **Run an Experiment**  
   ```bash
   python src/main.py -exp_nb 1
   ```
   Replace `1` with your desired experiment number.

3. **View Output**  
   After running, you'll find:
   - Tweet embeddings in `embeddings/`
   - Adjacency matrices in `adjacency_matrices/`
   - Training plots in `plots/`

---

## 📊 Datasets

The framework supports multiple datasets for extremist narrative analysis:

| Dataset | Description | Filename | Code Reference |
|---------|-------------|----------|----------------|
| **Toxigen** | English toxicity and stereotype analysis | `Toxigen.csv` | `ToxigenDataset` |
| **FRENK LGBTEn** | English LGBT-related posts and annotations | `LGBTEn.csv` | `LGBTEnDataset` |
| **FRENK MigrantsEn** | English migrant-related posts and annotations | `MigrantsEn.csv` | `MigrantsEnDataset` |
| **Multilingual EN Corpus German** | 1000 German tweets with extremism annotations | `Multilingual_EN_Corpus_DATA_GERMAN.xlsx` | `MultilingualENCorpusGermanDataset` |
| **Multilingual EN Corpus French** | 1000 French tweets with extremism annotations | `Multilingual_EN_Corpus_DATA_FRENCH.xlsx` | `MultilingualENCorpusFrenchDataset` |
| **Multilingual EN Corpus Cypriot** | 1000 Cypriot tweets with extremism annotations | `Multilingual_EN_Corpus_DATA_CYPRIOT.xlsx` | `MultilingualENCorpusCypriotDataset` |
| **Multilingual EN Corpus Slovene** | 1000 Slovene tweets with extremism annotations | `Multilingual_EN_Corpus_DATA_SLOVENE.xlsx` | `MultilingualENCorpusSloveneDataset` |

> **💡 Tip:** Use the code reference names in `preprocessing.py` when loading datasets programmatically.

### Dataset-Specific Parameters

When running experiments with different datasets, you need to adjust the following parameters:

| Dataset | exp_nb | Text Column | Label Column(s) | Context Columns |
|---------|--------|-------------|-----------------|-----------------|
| **Toxigen** | 1 | `text` | `target_group` | `In-Group`, `Out-group` |
| **FRENK LGBTEn** | 2 | `text` | `annotation_type` | `In-Group`, `Out-group` |
| **FRENK MigrantsEn** | 3 | `text` | `annotation_type` | `In-Group`, `Out-group` |
| **Multilingual EN Corpus German** | 4 | `Text` | `Topic`, `Initiating Problem`, `Intolerance`, `Hostility to out-group (e.g. verbal attacks, belittlement, instillment of fear, incitement to violence)`, `Polarization/Othering`, `Perceived Threat`, `Solution` | `In-Group`, `Out-group` |
| **Multilingual EN Corpus French** | 5 | `Text` | Same as German | `In-Group`, `Out-group` |
| **Multilingual EN Corpus Cypriot** | 6 | `Tweet, text` | Same as German + `Tone of Post`, `Narrator`, etc. | `In-Group`, `Out-group` |
| **Multilingual EN Corpus Slovene** | 7 | `Tweet, Text` | Same as German | `In-Group`, `Out-group` |

#### How to Adapt Examples for Different Datasets

All examples in this README use **Example 1 (Toxigen)** by default. To use a different dataset:

1. **Change the experiment number (`-exp_nb`)**: Use the corresponding `exp_nb` from the table above
2. **Update the dataset class**: Replace `ToxigenDataset` with the appropriate class name
3. **Adjust the label column**: Use the correct label column(s) from the table above
4. **Update the text column**: Use the correct text column name (usually `text` or `Text`)
5. **Keep or adjust context columns**: Most datasets use `In-Group` and `Out-group`, but some have additional context columns


---

## 🧠 Methodology

### Contrastive Loss

**Purpose:**  
Ensure that representations (embeddings) of the same node under different graph views are similar, while representations of different nodes are distinct.

**How it works:**

1. **Create Two Views** – For each node, generate two versions with small random changes (augmentations)
2. **Measure Similarity** – Calculate cosine similarity between views (1 = identical, -1 = opposite)
3. **Maximize** – Similarity for the same node across views (positive pairs)
4. **Minimize** – Similarity for different nodes (negative pairs)

#### 📐 Mathematical Formulation

<img src="images/contrastive.png" alt="L_total equation" width="400"/>

Where:
- `sim(·, ·)` is cosine similarity
- `τ` is the temperature parameter (controls discrimination)
- The loss encourages similar embeddings for positive pairs and dissimilar embeddings for negative pairs

---

## 📖 Usage Guide

### Running Experiments

Below are the main experiments with dataset mappings:

| Exp # | Dataset | Configuration | Command |
|-------|---------|---------------|---------|
| 1 | Toxigen (English) | `exp_nb=1`, `epoch=4000` | `python src/main.py -exp_nb 1` |
| 2 | FRENK LGBTEn | `exp_nb=2`, `epoch=4000` | `python src/main.py -exp_nb 2` |
| 3 | FRENK MigrantsEn | `exp_nb=3`, `epoch=4000` | `python src/main.py -exp_nb 3` |
| 4 | Multilingual EN Corpus German | `exp_nb=4`, `epoch=4000` | `python src/main.py -exp_nb 4` |
| 5 | Multilingual EN Corpus French | `exp_nb=5`, `epoch=4000` | `python src/main.py -exp_nb 5` |
| 6 | Multilingual EN Corpus Cypriot | `exp_nb=6`, `epoch=4000` | `python src/main.py -exp_nb 6` |
| 7 | Multilingual EN Corpus Slovene | `exp_nb=7`, `epoch=4000` | `python src/main.py -exp_nb 7` |

> **💡 Tip:** Adjust epochs and other parameters in `experiment_params.csv` for your specific needs.

------

### Adding Context-Based Edges

This framework supports **context-guided graph structure learning**, extending the SUBLIME paradigm. Context attributes (e.g., *In-Group*, *Out-group*) guide the learning of feature-based graph structures via contrastive learning.

#### How It Works

Graph structure learning operates on **two graph views**:

- **🔗 Anchor View (Context Graph)**  
  A fixed adjacency matrix constructed from shared context attribute values.  
  Nodes sharing the same attribute are connected.  
  This graph is **not learned** – it serves as a structural guide.

- **🧩 Learner View (Feature Graph)**  
  A learnable adjacency matrix inferred from node features (text embeddings).  
  Uses cosine similarity and is refined during training.

A contrastive loss aligns node representations from both views, encouraging the learned feature-based structure to be consistent with contextual relationships.

After training, a **k-nearest neighbor (kNN)** graph is constructed from the learned embeddings for downstream tasks.

#### Configuration Parameters

| Parameter | Description |
|-----------|-------------|
| `--use_context_adj` | Activate context-based adjacency as anchor view |
| `--add_attr_edges` | Construct context-based anchor graph |
| `--context_columns` | Specify which context columns to use (e.g., "In-Group" "Out-group") |
| `--attr_edges_max` | Limit maximum context edges per node |

#### Example Command

```bash
python src/main.py -exp_nb 1 --gpu 0 \
  --use_context_adj \
  --add_attr_edges \
  --context_columns "In-Group" "Out-group" \
  --attr_edges_max 10 \
  --epochs 4000 \
  --lr 0.005 --w_decay 1e-4 \
  --nlayers 2 --hidden_dim 256 \
  --rep_dim 256 --proj_dim 128 \
  --dropout 0.5 --dropedge_rate 0.2
```

---

### Sensitivity Analysis with Sweeps

Explore the impact of different parameters on graph construction and model performance through systematic parameter sweeps.

#### Sweep Parameters

- **`k` (text-based edges):** Number of neighbors connected using text embeddings
- **`attr_edges_max` (context-based edges):** Maximum context-driven edges per node

#### Example Command

```bash
python src/main.py -exp_nb 1 --gpu 0 --run_sweep \
  --sweep_k_list 5,10,15,25 \
  --sweep_max_list 0,10,50 \
  --context_columns "In-Group" "Out-group" \
  --use_context_adj \
  --add_attr_edges \
  --sweep_epochs 400 \
  --epochs 400 --maskfeat_rate_anchor 0.35 --maskfeat_rate_learner 0.35 \
  --temperature 0.08 --lr 0.005 --w_decay 1e-4 \
  --nlayers 2 --hidden_dim 256 --rep_dim 256 --proj_dim 128
```

> **💡 Tip:** Sweep results are saved for batch processing with node classification tasks.

---

### Node Classification

Run node classification experiments on the learned graph representations.

#### Single Experiment

```bash
python src/node_cls.py \
  --cpu_only \
  --dataset Toxigen \
  --label_col "target_group" \
  --adjacency_matrix /abs/path/to/adjacency_file.pkl \
  --embeddings_file /abs/path/to/embeddings_file.npy \
  [--feature_cols "In-Group" "Out-group"] \
  --epochs 600 \
  --hidden_dim 256 \
  --num_layers 2 \
  --dropout 0.5
```

#### Batch Mode

Process multiple node classification experiments using a CSV file (e.g., from sweep results):

```bash
nohup python src/run_node_cls_batch.py \
  ./sweep_results_main_toxigen.csv \
  "target_group" \
  Toxigen \
  1 > LOG_toxigen.log 2>&1 &
```

> **📝 Note:** The batch mode allows classification for multiple targets simultaneously.

---


### BERT Baseline Text Classification

Compare graph-based approaches against a baseline text classification model using BERT.

#### Example Command

```bash
nohup python src/bert_text_classification_features.py \
  --data_path ./datasets/Toxigen.csv \
  --text_col text \
  --targets "target_group" \
  --feature_cols "In-Group" "Out-group" \
  --epochs 5 \
  --batch_size 16 \
  --lr 2e-5 \
  --max_length 128 \
  --seeds 0 21 42 84 123 \
  --split_ratios 0.6 0.2 0.2 \
  --gpu 0 > bert_toxigen.log 2>&1 &
```

> **💡 Tip:** This script supports additional feature columns (context attributes) alongside text classification.

---

## 📦 Output Files & Embeddings

The pipeline generates three types of embeddings at different stages:

### Embedding Types

| Type | Filename Pattern | Description | When Generated |
|------|------------------|-------------|----------------|
| **Text Embeddings** | `{dataset}_embeddings_exp{N}.npy` | Raw semantic embeddings from sentence transformer (`paraphrase-multilingual-MiniLM-L12-v2`) | Dataset initialization |
| **Combined Embeddings** | `combined_exp{N}_max{M}.npy` | Text embeddings + one-hot encoded context attributes | Before training (with `--run_sweep`) |
| **Learned Embeddings** | `embeddings__exp{N}__ctxcols_{cols}__ntrials_{T}.npy` | **Final embeddings** refined by graph contrastive learning | After training completes |

### Which Embeddings Should I Use?

- **For downstream tasks** (node classification, clustering): Use **learned embeddings** 
- **For ablation studies**: Compare text vs. combined vs. learned
- **For graph construction**: Text or combined embeddings are used as input features

> **💡 Tip:** Learned embeddings capture both semantic content and graph structure, making them the most powerful representation for narrative analysis.

---


## ⚙️ Configuration

### Hardware & Reproducibility

| Parameter | Type | Description |
|-----------|------|-------------|
| `--gpu` | int | GPU index to use (default: 0). Falls back to CPU if unavailable |
| `--ntrials` | int | Number of repeated experiments for reproducibility |

### Contrastive Learning (GCL)

| Parameter | Type | Description |
|-----------|------|-------------|
| `--temperature` | float | Temperature parameter for contrastive loss (controls discrimination) |
| `--maskfeat_rate_anchor` | float | Feature masking rate for anchor view |
| `--maskfeat_rate_learner` | float | Feature masking rate for learner view |
| `--nlayers` | int | Number of GCN encoder layers |
| `--hidden_dim` | int | Hidden dimension size in encoder |
| `--rep_dim` | int | Final representation (embedding) dimension |
| `--proj_dim` | int | Projector head dimension after encoder |
| `--dropout` | float | Dropout rate inside encoder/projector |
| `--dropedge_rate` | float | Edge dropout rate for graph augmentation |

### Graph Learner & Adjacency

| Parameter | Type | Description |
|-----------|------|-------------|
| `--sparse` | bool | Use sparse adjacency operations |
| `--type_learner` | str | Graph learner type: `fgp`, `mlp`, `att`, `gnn` |
| `--k` | int | Number of neighbors for k-NN when building adjacency |
| `--sim_function` | str | Similarity function: `cosine`, `dot` |
| `--activation_learner` | str | Activation used in graph learner (e.g., `relu`) |
| `--gsl_mode` | str | Graph structure mode: `structure_refinement`, `structure_inference` |
| `--n_neighbors` | int | Number of neighbors to keep in final adjacency |
| `--sym` | bool | Symmetrize the adjacency matrix |

### Training & Optimization

| Parameter | Type | Description |
|-----------|------|-------------|
| `--epochs` | int | Number of training epochs |
| `--lr` | float | Learning rate |
| `--w_decay` | float | Weight decay (L2 regularization) |

---

## 🔧 Troubleshooting

### Common Issues

#### DGL Installation Errors
**Problem:** Error installing or using DGL on GPU  
**Solution:** The default installation is CPU-only. For GPU support:
```bash
pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html  # For CUDA 11.8
```
Refer to [DGL installation guide](https://www.dgl.ai/pages/start.html) for other CUDA versions.

#### Out of Memory Errors
**Problem:** CUDA out of memory during training  
**Solution:**
- Reduce `--hidden_dim`, `--rep_dim`, or `--proj_dim`
- Decrease batch size (if applicable)
- Use `--cpu_only` flag for smaller datasets

#### Missing Dependencies
**Problem:** Import errors for specific packages  
**Solution:**
```bash
pip install -r requirements.txt --upgrade
```

#### Adjacency Matrix Not Found
**Problem:** Node classification fails with missing adjacency matrix  
**Solution:** Ensure you've run the main training script first:
```bash
python src/main.py -exp_nb <N>
```
This generates embeddings and adjacency matrices needed for downstream tasks.

### Getting Help

If you encounter issues not covered here:
1. Check the experiment logs in your output directory
2. Verify your `experiment_params.csv` configuration
3. Open an issue on the [GitHub repository](https://github.com/diniaouri/GR-EN-A-DE)

---
