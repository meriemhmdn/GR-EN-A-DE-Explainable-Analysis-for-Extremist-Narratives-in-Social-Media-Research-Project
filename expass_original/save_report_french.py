#!/usr/bin/env python3

import pandas as pd
import sys
from pathlib import Path
from analyze_edge_explanation_french import analyze_important_edges, load_toxigen_data

# Output directory
output_dir = Path("analysis_results")
output_dir.mkdir(exist_ok=True)

# =========================
# 1. SAVE FULL TEXT REPORT
# =========================

original_stdout = sys.stdout

with open(output_dir / "edge_analysis_full_report.txt", "w") as f:
    sys.stdout = f
    
    analyze_important_edges(
        edge_csv="edge_importance_exp4.csv",  # ⚠️ exp4 pour FR
        top_k=None,
        show_context=True,
    )

sys.stdout = original_stdout

print("✓ Full analysis saved")

# =========================
# 2. LOAD DATASET (FR VERSION)
# =========================

texts_df = load_toxigen_data()  # ✅ même fonction que ton analyse

# Nettoyage (IMPORTANT)
texts_df = texts_df.dropna(subset=["text"]).reset_index(drop=True)

# =========================
# 3. LOAD EDGES
# =========================

edges_df = pd.read_csv("edge_importance_exp4.csv")
edges_df = edges_df.sort_values(by="importance_score", ascending=False)
top_edges = edges_df

# =========================
# 4. BUILD ENRICHED CSV
# =========================

enriched_data = []

for _, row in top_edges.iterrows():
    source = int(row['source_node'])
    target = int(row['target_node'])
    
    source_row = texts_df.iloc[source] if source < len(texts_df) else None
    target_row = texts_df.iloc[target] if target < len(texts_df) else None
    
    enriched_data.append({
        'rank': int(row['rank']),
        'importance_score': row['importance_score'],
        'source_node': source,
        'target_node': target,
        
        'source_label': source_row.get('target_group') if source_row is not None else 'N/A',
        'target_label': target_row.get('target_group') if target_row is not None else 'N/A',
        
        'source_in_group': source_row.get('In-Group') if source_row is not None else 'N/A',
        'source_out_group': source_row.get('Out-group') if source_row is not None else 'N/A',
        
        'target_in_group': target_row.get('In-Group') if target_row is not None else 'N/A',
        'target_out_group': target_row.get('Out-group') if target_row is not None else 'N/A',
        
        'source_text': (source_row.get('text')[:200] if source_row is not None else 'N/A'),
        'target_text': (target_row.get('text')[:200] if target_row is not None else 'N/A'),
    })

# =========================
# 5. SAVE CSV
# =========================

enriched_df = pd.DataFrame(enriched_data)
enriched_df.to_csv(output_dir / "top_100_edges_with_context.csv", index=False)

print(f"\n✓ Analysis saved to:")
print(f"  📄 {output_dir}/edge_analysis_full_report_fr.txt")
print(f"  📊 {output_dir}/top_100_edges_with_context_fr.csv")