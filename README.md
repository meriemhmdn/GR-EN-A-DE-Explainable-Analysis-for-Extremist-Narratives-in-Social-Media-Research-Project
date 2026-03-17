# GRENADE-EXPASS Pipeline

> Interpretable Graph Learning for Text Analysis

## Quick Start

### 1. Install
```bash
git clone https://github.com/diniaouri/GRENADE-EXPASS-Pipeline.git
cd GRENADE-EXPASS-Pipeline
pip install -r requirements.txt
```

### 2. Run GRENADE (Learn Graph)
```bash
cd grenade_original/Contrastive_Learning_Approach
python src/main.py -exp_nb 1 --epochs 400
```



### Basic Mode (Text-based edges only)
```bash
cd grenade_original/Contrastive_Learning_Approach
python src/main.py -exp_nb 1 --epochs 4000
```
### Context-Guided Mode (Text + EN attribute edges)

```bash
cd grenade_original/Contrastive_Learning_Approach

python src/main.py -exp_nb 1 --gpu 0 \
  --use_context_adj \
  --add_attr_edges \
  --context_columns "In-Group" "Out-group" \
  --attr_edges_max 10 \
  --epochs 4000
```

Creates: `embeddings/embeddings__exp1__ntrials_1.npy` and `adjacency_matrices/adjacency_final__exp1__ntrials_1.pkl`

### 3. Run EXPASS (Train GNN + Explain)
```bash
cd ../../expass_original
python train.py \
  --grenade-embeddings ../grenade_original/Contrastive_Learning_Approach/embeddings/embeddings__exp1__ntrials_1.npy \
  --grenade-adjacency ../grenade_original/Contrastive_Learning_Approach/adjacency_matrices/adjacency_final__exp1__ntrials_1.pkl \
  --grenade-exp-nb 1 \
  --arch gcn \
  --explainer gnn_explainer \
  --epochs 150
```
Creates: `convergence_files/*.csv`, `grenade-gcn/*.pth`

### 4. Export Edge Importance
```bash
python export_explanations.py \
  --model grenade-gcn/loss-lrgnn_0.01-seed_912-best.pth \
  --embeddings ../grenade_original/Contrastive_Learning_Approach/embeddings/embeddings__exp1__ntrials_1.npy \
  --adjacency ../grenade_original/Contrastive_Learning_Approach/adjacency_matrices/adjacency_final__exp1__ntrials_1.pkl \
  --output edge_importance_exp1.csv \
  --arch gcn \
  --layers 3 \
  --nhid 32 \
  --num_classes 13

Creates: `edge_importance_exp1.csv``
```

### 5. Analyze Top Edges (with text) - TO ADAPT TO THE OTHER DATASET (CURRENTLY EXP1 SUPPORTED)
```bash
python analyze_edge_explanations_detailed.py \
  --edges edge_importance_exp1.csv \
  --top-k 50
```
Shows detailed analysis with actual text from nodes

### 6. Save Analysis Report - TO ADAPT TO THE OTHER DATASET (CURRENTLY EXP1 SUPPORTED )
```bash
python save_analysis_report.py
```
Creates: `analysis_results/edge_analysis_full_report.txt` and `analysis_results/top_100_edges_with_context.csv`



## Different Datasets

```bash
# Toxigen (hate speech)
python src/main.py -exp_nb 1 --epochs 4000

# LGBTEn narratives
python src/main.py -exp_nb 2 --epochs 4000

# MigrantsEn narratives
python src/main.py -exp_nb 3 --epochs 4000

# Multilingual EN Corpus FRENCH 
python src/main.py -exp_nb 4 --epochs 4000

# Multilingual EN Corpus GERMAN 
python src/main.py -exp_nb 5 --epochs 4000

#  Multilingual EN Corpus CYPRIOT 
python src/main.py -exp_nb 6 --epochs 4000

#  Multilingual EN Corpus SLOVENE
python src/main.py -exp_nb 7 --epochs 4000

```

## Different Architectures

```bash
# GCN architecture
python train.py --grenade-exp-nb 1 --arch gcn --explainer gnn_explainer --epochs 150

# GraphConv architecture
python train.py --grenade-exp-nb 1 --arch graphconv --explainer gnn_explainer --epochs 150

# With PGM Explainer
python train.py --grenade-exp-nb 1 --arch gcn --explainer pgmexplainer --epochs 150
```

## Output Files

| File | Description |
|------|-------------|
| `embeddings/embeddings_exp1_epoch400.npy` | Node features (N × 256) |
| `adjacency_matrices/adj_exp1_epoch400.pkl` | Graph structure (N × N sparse) |
| `edge_importance_exp1.csv` | Edge importance scores (ranked) |
| `convergence_files/loss-*.csv` | Training metrics per epoch |

## Your Own Dataset

1. **Create CSV** with `text` and `label` columns
2. **Add to** `grenade_original/Contrastive_Learning_Approach/src/experiment_params.csv`
3. **Run** `python src/main.py -exp_nb YOUR_EXP_NB --epochs 400`

## What's New

### Original Implementations
- **GRENADE**: Graph contrastive learning for text (from [GR-EN-A-DE repo](https://github.com/diniaouri/GR-EN-A-DE))
- **EXPASS**: Explanation-directed message passing for GNNs (from [EXPASS repo](https://github.com/AikyamLab/Expass))

### Our Contributions
- **GRENADE-EXPASS connector**: Load GRENADE outputs directly into EXPASS
- **Custom dataset loader** (`grenade_dataset.py`): Bridge between GRENADE embeddings/adjacency and PyTorch Geometric
- **Unified pipeline**: Run text → graph → explanations in one workflow
- **Direct path support**: Use pre-created embeddings/adjacency files with `--grenade-embeddings` and `--grenade-adjacency` flags

##  Pipeline Architecture

### **Stage 1: GRENADE** 
> **Graph Representation Learning**

| Component | Details |
|-----------|---------|
| **Input** | Toxigen dataset (8,000+ texts with metadata) |
| **Process** | • Build similarity graph<br>• Apply contrastive learning<br>• Generate embeddings |
| **Output** | • `embeddings__exp1__ntrials_1.npy`<br>• `adjacency_final__exp1__ntrials_1.pkl` |



### **Stage 2: EXPASS** 
> **Edge Importance Explanation**

| Component | Details |
|-----------|---------|
| **Input** | • GRENADE embeddings<br>• GRENADE adjacency matrix<br>• Original dataset labels |
| **Process** | • Convert to PyTorch Geometric format<br>• Train GNN classifier<br>• Apply GNNExplainer |
| **Output** | • `edge_importance_exp1.csv`<br>• Trained model `.pth` |



### **Stage 3: Analysis** 
> **Human-Readable Interpretations**

| Component | Details |
|-----------|---------|
| **Input** | • Edge importance scores<br>• Original text content<br>• Metadata (In-Group/Out-group) |
| **Process** | • Join edge scores with text<br>• Rank by importance<br>• Generate reports |
| **Output** | • `top_100_edges_with_context.csv`<br>• `edge_analysis_full_report.txt` |
## Citation

```bibtex

@article{expass2022,
  title={Towards Training GNNs using Explanation Directed Message Passing},
  author={Longa, Antonio and Azzolin, Steve and Santin, Gabriele and Lió, Pietro and Lepri, Bruno and Passerini, Andrea},
  journal={Learning on Graphs Conference (LoG)},
  year={2022}
}
```

## License

MIT License - see repository for details
