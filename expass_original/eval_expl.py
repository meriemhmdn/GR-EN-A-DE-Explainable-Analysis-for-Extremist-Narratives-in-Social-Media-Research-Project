'''import pandas as pd
import numpy as np

# Fichier input
file_path = "analysis_results/narrative_explanations_all_models.csv"
df = pd.read_csv(file_path)

models = ["deepseek", "llama", "mistral"]

# ---------------------------
# FONCTIONS D'ÉVALUATION
# ---------------------------

def contains(text, keyword):
    if pd.isna(text) or pd.isna(keyword):
        return 0
    return int(str(keyword).lower() in str(text).lower())

def word_count(text):
    if pd.isna(text):
        return 0
    return len(str(text).split())

def sentence_count(text):
    if pd.isna(text):
        return 0
    return str(text).count('.') + str(text).count('!') + str(text).count('?')

def has_relation_words(text):
    keywords = ["because", "due to", "therefore", "this suggests", "connection", "relationship"]
    return int(any(k in str(text).lower() for k in keywords))

def has_group_mention(text, source_label, target_label):
    return max(
        contains(text, source_label),
        contains(text, target_label)
    )

def has_in_group(text, in_group):
    return contains(text, in_group)

def has_out_group(text, out_group):
    return contains(text, out_group)

# ---------------------------
# ÉVALUATION
# ---------------------------

results = []

for model in models:
    print(f"\nEvaluating {model}...")

    group_scores = []
    relation_scores = []
    in_group_scores = []
    out_group_scores = []
    length_scores = []
    sentence_scores = []

    for _, row in df.iterrows():
        text = row[f"explanation_{model}"]

        # Coverage
        group = has_group_mention(text, row["source_label"], row["target_label"])
        relation = has_relation_words(text)
        in_group = has_in_group(text, row["source_in_group"])
        out_group = has_out_group(text, row["source_out_group"])

        # Structure
        length = word_count(text)
        sentences = sentence_count(text)

        group_scores.append(group)
        relation_scores.append(relation)
        in_group_scores.append(in_group)
        out_group_scores.append(out_group)
        length_scores.append(length)
        sentence_scores.append(sentences)

    # Moyennes
    results.append({
        "model": model,
        "group_coverage": np.mean(group_scores),
        "relation_score": np.mean(relation_scores),
        "in_group_coverage": np.mean(in_group_scores),
        "out_group_coverage": np.mean(out_group_scores),
        "avg_word_count": np.mean(length_scores),
        "avg_sentences": np.mean(sentence_scores)
    })

# ---------------------------
# RÉSULTATS
# ---------------------------

results_df = pd.DataFrame(results)

print("\n" + "="*60)
print("EVALUATION RESULTS")
print("="*60)
print(results_df)

# Sauvegarde
results_df.to_csv("analysis_results/llm_evaluation_scores.csv", index=False)

print("\nSaved to analysis_results/llm_evaluation_scores.csv")'''


import pandas as pd
import numpy as np
import re

# ---------------------------
# FICHIER INPUT
# ---------------------------
file_path = "analysis_results/narrative_explanations_all_models_1.csv"
df = pd.read_csv(file_path)

models = ["deepseek", "llama", "mistral"]

# ---------------------------
# FONCTIONS D'ÉVALUATION AMÉLIORÉES
# ---------------------------

def contains_keyword(text, keyword):
    """Renvoie 1 si le mot/label est présent, 0 sinon."""
    if pd.isna(text) or pd.isna(keyword):
        return 0
    return 1 if str(keyword).lower() in str(text).lower() else 0

def word_count(text):
    if pd.isna(text):
        return 0
    return len(str(text).split())

def sentence_count(text):
    if pd.isna(text):
        return 0
    return str(text).count('.') + str(text).count('!') + str(text).count('?')

def has_relation_words(text):
    """Renvoie 1 si au moins un mot de relation est présent."""
    if pd.isna(text):
        return 0
    keywords = ["because", "due to", "therefore", "this suggests", "connection", "relationship"]
    text = str(text).lower()
    return 1 if any(k in text for k in keywords) else 0

def has_group_mention(text, source_label, target_label):
    """1 si au moins un label source ou target est présent"""
    return max(contains_keyword(text, source_label), contains_keyword(text, target_label))

def has_in_group(text, in_group):
    return contains_keyword(text, in_group)

def has_out_group(text, out_group):
    return contains_keyword(text, out_group)

# ---------------------------
# ÉVALUATION
# ---------------------------

results = []

for model in models:
    print(f"\nEvaluating {model}...")

    group_scores = []
    relation_scores = []
    in_group_scores = []
    out_group_scores = []
    length_scores = []
    sentence_scores = []

    for _, row in df.iterrows():
        text = row[f"explanation_{model}"]

        # Coverage
        group = has_group_mention(text, row["source_label"], row["target_label"])
        relation = has_relation_words(text)
        in_group = has_in_group(text, row["source_in_group"])
        out_group = has_out_group(text, row["source_out_group"])

        # Structure
        length = word_count(text)
        sentences = sentence_count(text)

        group_scores.append(group)
        relation_scores.append(relation)
        in_group_scores.append(in_group)
        out_group_scores.append(out_group)
        length_scores.append(length)
        sentence_scores.append(sentences)

    # Moyennes
    results.append({
        "model": model,
        "group_coverage": np.mean(group_scores),
        "relation_score": np.mean(relation_scores),
        "in_group_coverage": np.mean(in_group_scores),
        "out_group_coverage": np.mean(out_group_scores),
        "avg_word_count": np.mean(length_scores),
        "avg_sentences": np.mean(sentence_scores)
    })

# ---------------------------
# RÉSULTATS
# ---------------------------

results_df = pd.DataFrame(results)

print("\n" + "="*60)
print("EVALUATION RESULTS")
print("="*60)
print(results_df)

# Sauvegarde
results_df.to_csv("analysis_results/llm_evaluation_score_new_1.csv", index=False)
print("\nSaved to analysis_results/llm_evaluation_scores_new_1.csv")