import pandas as pd
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

print("Narrative explanations saved to:", output_file)