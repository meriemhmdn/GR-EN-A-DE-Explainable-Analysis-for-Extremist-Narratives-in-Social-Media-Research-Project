'''import pandas as pd
import requests
from tqdm import tqdm

input_file = "analysis_results/top_100_edges_with_context.csv"
output_file = "analysis_results/narrative_explanations.csv"

df = pd.read_csv(input_file)

narratives = []

for _, row in tqdm(df.iterrows(), total=len(df)):

    prompt = f"""
A Graph Neural Network detected a possible extremist narrative.

Edge importance score: {row['importance_score']}

Source message:
{row['source_text']}

Target message:
{row['target_text']}

Source narrative attributes:
Label: {row['source_label']}
In-group: {row['source_in_group']}
Out-group: {row['source_out_group']}

Target narrative attributes:
Label: {row['target_label']}
In-group: {row['target_in_group']}
Out-group: {row['target_out_group']}

Explain in natural language why the connection between these two
messages may contribute to an extremist or hostile narrative.

Focus on:
- the narrative relation between the messages
- the role of in-group / out-group framing
- why the model might consider this connection important.
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "deepseek-r1:7b",
            "prompt": prompt,
            "stream": False
        }
    )

    narrative = response.json()["response"]
    narratives.append(narrative)

df["graphxain_explanation"] = narratives

df.to_csv(output_file, index=False)

print("Narrative explanations saved to:", output_file)'''


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

        '''        prompt = f"""
        A Graph Neural Network detected a potentially extremist narrative connection between two messages.

        Edge importance score: {row['importance_score']}

        Source message:
        {row['source_text']}

        Target message:
        {row['target_text']}

        Source attributes:
        - Label: {row['source_label']}
        - In-group: {row['source_in_group']}
        - Out-group: {row['source_out_group']}

        Target attributes:
        - Label: {row['target_label']}
        - In-group: {row['target_in_group']}
        - Out-group: {row['target_out_group']}

        Task:
        Write a concise explanation (3-4 sentences) describing why this connection is important.

        Requirements:
        - Mention the relationship between the two messages
        - Explicitly refer to the targeted group (if any)
        - Explain the role of in-group / out-group framing
        - Justify why the connection may be considered important by the model

        Do not use bullet points. Write a clear and coherent paragraph.
        """'''
        prompt = f"""
            A Graph Neural Network identified an important connection between two messages in a potential extremist narrative.

            Edge importance score: {row['importance_score']}

            Source message:
            {row['source_text']}

            Target message:
            {row['target_text']}

            Source attributes:
            - Label: {row['source_label']}
            - In-group: {row['source_in_group']}
            - Out-group: {row['source_out_group']}

            Target attributes:
            - Label: {row['target_label']}
            - In-group: {row['target_in_group']}
            - Out-group: {row['target_out_group']}

            Task:
            Analyze the relationship between the two messages and provide a structured explanation.

            You must identify:
            - the actors involved
            - the relationship between the messages (e.g., agreement, reinforcement, opposition, escalation)
            - the narrative type (e.g., hostility, victimization, blame, fear, identity reinforcement)
            - why this connection is important for the model prediction

            Guidelines:
            - Consider semantic similarity or reinforcement
            - Consider in-group / out-group dynamics
            - Consider how the messages may contribute to extremist discourse

            Output format (STRICT):
            Return ONLY a valid JSON object with the following structure:

            {{
            "actors": "...",
            "relationship": "...",
            "narrative_type": "...",
            "importance": "..."
            }}

            Constraints:
            - The output MUST be valid JSON
            - Do not add any text before or after the JSON
            - Do not use markdown formatting
            - Use complete and concise sentences inside each field
            - Base your explanation ONLY on the provided messages and attributes
            """

        narrative = generate(prompt, model_name)
        narratives.append(narrative)

    df[f"explanation_{model_key}"] = narratives

# Sauvegarde finale avec tous les modèles
output_file = "analysis_results/narrative_explanations_all_models_exp2.csv"
df.to_csv(output_file, index=False)

print("Saved:", output_file)