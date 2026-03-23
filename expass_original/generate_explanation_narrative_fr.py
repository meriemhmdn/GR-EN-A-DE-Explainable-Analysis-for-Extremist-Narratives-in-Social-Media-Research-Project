


import pandas as pd
import requests
from tqdm import tqdm

input_file = "analysis_results/top_100_edges_with_context.csv"
df = pd.read_csv(input_file)

models = {
    "deepseek": "deepseek-r1:7b",
    "llama": "llama3",
    "mistral": "mistral"
}

def generate(prompt, model_name):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model_name,
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]

for model_key, model_name in models.items():
    print(f"\nGenerating with {model_key}...\n")

    narratives = []

    for _, row in tqdm(df.iterrows(), total=len(df)):

        prompt = f"""
            Un modèle de type Graph Neural Network a identifié une connexion importante entre deux messages dans un récit potentiellement extrémiste.

            Score d'importance de l'arête : {row['importance_score']}

            Message source :
            {row['source_text']}

            Message cible :
            {row['target_text']}

            Attributs du message source :
            - Thématique : {row['source_label']}
            - In-group : {row['source_in_group']}
            - Out-group : {row['source_out_group']}

            Attributs du message cible :
            - Thématique : {row['target_label']}
            - In-group : {row['target_in_group']}
            - Out-group : {row['target_out_group']}

            Tâche :
            Analyse la relation entre ces deux messages et fournis une explication structurée.

            Tu dois identifier :
            - les acteurs ou groupes impliqués
            - la relation entre les messages (ex : accord, renforcement, opposition, escalade)
            - le type de récit (ex : hostilité, victimisation, accusation, peur, renforcement identitaire)
            - pourquoi cette connexion est importante pour la prédiction du modèle

            Consignes :
            - Prends en compte la similarité sémantique ou le renforcement entre les messages
            - Analyse la dynamique in-group / out-group
            - Explique comment ces messages peuvent contribuer à un discours hostile ou extrémiste

            Format de sortie (STRICT) :
            Retourne UNIQUEMENT un objet JSON valide avec la structure suivante :

            {{
            "actors": "...",
            "relationship": "...",
            "narrative_type": "...",
            "importance": "..."
            }}

            Contraintes :
            - La sortie DOIT être un JSON valide
            - Ne pas ajouter de texte avant ou après le JSON
            - Ne pas utiliser de format markdown
            - Utiliser des phrases complètes et concises dans chaque champ
            - Se baser UNIQUEMENT sur les messages et attributs fournis
            """

        narrative = generate(prompt, model_name)
        narratives.append(narrative)

    df[f"explanation_{model_key}"] = narratives

# Sauvegarde finale avec tous les modèles
output_file = "analysis_results/narrative_explanations_all_models_fr.csv"
df.to_csv(output_file, index=False)

print("Saved:", output_file)






