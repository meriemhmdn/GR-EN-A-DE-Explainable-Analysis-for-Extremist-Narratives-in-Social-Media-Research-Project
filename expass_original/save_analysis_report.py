#!/usr/bin/env python3
"""
Save the edge analysis to readable files.
"""

import pandas as pd
import sys
from pathlib import Path


# Redirect output to file
output_dir = Path("analysis_results")
output_dir.mkdir(exist_ok=True)

# Run the analysis and save to file
sys.stdout = open(output_dir / "edge_analysis_full_report.txt", "w")

# Import and run the analysis
exec(open("analyze_edge_explanations_detailed.py").read().replace('if __name__ == "__main__":', 'if False:'))

analyze_important_edges(
    edge_csv="edge_importance_exp1.csv",
    top_k=50,
    show_context=True,
)

sys.stdout.close()

# Also create a summary CSV with top edges + text
edges_df = pd.read_csv("edge_importance_exp1.csv")
texts_df = pd.read_csv("../grenade_original/Contrastive_Learning_Approach/datasets/Toxigen.csv")

top_edges = edges_df.head(100)

# Create enriched CSV
enriched_data = []
for idx, row in top_edges.iterrows():
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
        'source_text': source_row.get('text')[:200] if source_row is not None else 'N/A',
        'target_text': target_row.get('text')[:200] if target_row is not None else 'N/A',
    })

enriched_df = pd.DataFrame(enriched_data)
enriched_df.to_csv(output_dir / "top_100_edges_with_context.csv", index=False)

print(f"\n✓ Analysis saved to:")
print(f"  📄 {output_dir}/edge_analysis_full_report.txt")
print(f"  📊 {output_dir}/top_100_edges_with_context.csv")
