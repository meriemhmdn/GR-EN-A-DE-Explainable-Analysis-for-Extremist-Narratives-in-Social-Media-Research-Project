import pandas as pd
import requests
from tqdm import tqdm
from collections import defaultdict

# --- Configurations ---
input_file = "analysis_results/top_100_edges_with_context.csv"
output_folder = "analysis_results/"
k = 3  # longueur des chaînes de noeuds
max_nodes = 10  # nombre de noeuds à traiter

models = {
    "deepseek": "deepseek-r1:7b",
    "llama": "llama3",
    "mistral": "mistral"
}

# --- Chargement du DataFrame ---
df = pd.read_csv(input_file)

# --- Génération via API ---
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

# --- Construction du graphe ---
graph = defaultdict(list)
for _, row in df.iterrows():
    graph[row['source_text']].append(row['target_text'])

# --- Fonction DFS simple pour une chaîne de longueur k ---
def get_chain(graph, start_node, k):
    chain = [start_node]
    while len(chain) < k:
        last_node = chain[-1]
        if last_node not in graph or not graph[last_node]:
            break
        next_node = graph[last_node][0]  # premier voisin
        if next_node in chain:
            break
        chain.append(next_node)
    return chain

# --- Construction du prompt pour une chaîne ---
def build_prompt(chain, df):
    prompt_messages = []
    for node in chain:
        row = df[(df['source_text'] == node) | (df['target_text'] == node)].iloc[0]
        label = row['source_label'] if row['source_text'] == node else row['target_label']
        in_group = row['source_in_group'] if row['source_text'] == node else row['target_in_group']
        out_group = row['source_out_group'] if row['source_text'] == node else row['target_out_group']
        prompt_messages.append(f"""
Message :
{node}

Attributs :
- Thématique : {label}
- In-group : {in_group}
- Out-group : {out_group}
""")
    prompt_text = "\n".join(prompt_messages)

    prompt = f"""
Un modèle de type Graph Neural Network a identifié une chaîne de {len(chain)} messages interconnectés dans un récit potentiellement extrémiste.

{prompt_text}

Tâche :
Analyse la chaîne de messages et fournis une explication structurée.

Tu dois identifier :
- les acteurs ou groupes impliqués
- la dynamique narrative entre les messages (ex : continuité, renforcement, opposition)
- le type de récit (ex : hostilité, victimisation, accusation, peur, renforcement identitaire)
- pourquoi cette chaîne est importante pour la prédiction du modèle

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
- Utiliser des phrases complètes et concises dans chaque champ
- Se baser UNIQUEMENT sur les messages et attributs fournis
"""
    return prompt

# --- Génération des explications pour chaque modèle ---
for model_key, model_name in models.items():
    print(f"\nGenerating explanations with {model_key}...\n")
    narratives = []

    # Limiter aux 10 premiers noeuds
    nodes_to_process = list(graph.keys())[:max_nodes]

    for node in tqdm(nodes_to_process):
        chain = get_chain(graph, node, k)
        prompt = build_prompt(chain, df)
        narrative = generate(prompt, model_name)
        narratives.append({"start_node": node, "chain": chain, "explanation": narrative})

    # Sauvegarde
    output_file = f"{output_folder}narrative_explanations_chain_{model_key}.csv"
    pd.DataFrame(narratives).to_csv(output_file, index=False)
    print(f"Saved: {output_file}")