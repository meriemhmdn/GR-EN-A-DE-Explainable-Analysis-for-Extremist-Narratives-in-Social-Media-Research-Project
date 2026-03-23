'''import pandas as pd
import argparse

# Parser des arguments
parser = argparse.ArgumentParser(description="Afficher une explication GraphXAIN")
parser.add_argument(
    "--index",
    type=int,
    default=0,
    help="Index de la ligne à afficher (par défaut: 0)"
)

args = parser.parse_args()
index = args.index

# Charger le fichier
file_path = "analysis_results/narrative_explanations.csv"
df = pd.read_csv(file_path)

# Vérification de l'index
if index < 0 or index >= len(df):
    print(f"Erreur: index {index} hors limites (0 - {len(df)-1})")
    exit()

# Récupérer la ligne
row = df.iloc[index]

print("\n" + "="*80)
print("GRAPHXAIN NARRATIVE EXPLANATION")
print("="*80)

print(f"\nIndex: {index}")
print(f"Rank: {row['rank']}")
print(f"Edge Importance Score: {row['importance_score']:.4f}")

print("\n--- SOURCE MESSAGE ---")
print(row['source_text'])

print("\nLabel:", row['source_label'])
print("In-group:", row['source_in_group'])
print("Out-group:", row['source_out_group'])

print("\n--- TARGET MESSAGE ---")
print(row['target_text'])

print("\nLabel:", row['target_label'])
print("In-group:", row['target_in_group'])
print("Out-group:", row['target_out_group'])

print("\n--- GRAPHXAIN EXPLANATION ---")
print(row['graphxain_explanation'])

print("\n" + "="*80)'''
import pandas as pd
import argparse

# Parser des arguments
parser = argparse.ArgumentParser(description="Afficher une explication GraphXAIN multi-modèles")
parser.add_argument(
    "--index",
    type=int,
    default=0,
    help="Index de la ligne à afficher (par défaut: 0)"
)

args = parser.parse_args()
index = args.index

# Charger le fichier (IMPORTANT : nouveau fichier)
file_path = "analysis_results/narrative_explanations_all_models.csv"
df = pd.read_csv(file_path)

# Vérification de l'index
if index < 0 or index >= len(df):
    print(f"Erreur: index {index} hors limites (0 - {len(df)-1})")
    exit()

# Récupérer la ligne
row = df.iloc[index]

print("\n" + "="*100)
print("GRAPHXAIN MULTI-MODEL EXPLANATION")
print("="*100)

print(f"\nIndex: {index}")
print(f"Rank: {row['rank']}")
print(f"Edge Importance Score: {row['importance_score']:.4f}")

# SOURCE
print("\n--- SOURCE MESSAGE ---")
print(row['source_text'])

print("\nLabel:", row['source_label'])
print("In-group:", row['source_in_group'])
print("Out-group:", row['source_out_group'])

# TARGET
print("\n--- TARGET MESSAGE ---")
print(row['target_text'])

print("\nLabel:", row['target_label'])
print("In-group:", row['target_in_group'])
print("Out-group:", row['target_out_group'])

# EXPLANATIONS
print("\n" + "="*100)
print("EXPLANATIONS COMPARISON")
print("="*100)

print("\n--- DeepSeek ---")
print(row.get('explanation_deepseek', "N/A"))

print("\n--- LLaMA ---")
print(row.get('explanation_llama', "N/A"))

print("\n--- Mistral ---")
print(row.get('explanation_mistral', "N/A"))

print("\n" + "="*100)