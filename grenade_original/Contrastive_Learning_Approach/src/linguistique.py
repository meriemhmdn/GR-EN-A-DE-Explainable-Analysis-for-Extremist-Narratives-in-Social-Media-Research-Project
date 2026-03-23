import spacy
import numpy as np

nlp = spacy.load("en_core_web_sm")

def extract_linguistic_features(texts):
    feats = []
    
    for doc in nlp.pipe(texts, disable=["ner"]):
        tokens = [t for t in doc if not t.is_space]
        n_total = len(tokens)

        if n_total == 0:
            feats.append([0]*10)
            continue

        # POS counts
        n_nouns = sum(1 for t in tokens if t.pos_ == "NOUN")
        n_verbs = sum(1 for t in tokens if t.pos_ == "VERB")
        n_adjs  = sum(1 for t in tokens if t.pos_ == "ADJ")
        n_adv   = sum(1 for t in tokens if t.pos_ == "ADV")
        n_pron  = sum(1 for t in tokens if t.pos_ == "PRON")

        # Style features
        avg_word_len = np.mean([len(t.text) for t in tokens])
        sentence_len = n_total
        lexical_diversity = len(set([t.lemma_ for t in tokens])) / n_total

        # Stopwords ratio
        stop_ratio = sum(1 for t in tokens if t.is_stop) / n_total

        # Punctuation ratio
        punct_ratio = sum(1 for t in tokens if t.is_punct) / n_total

        feats.append([
            n_nouns / n_total,
            n_verbs / n_total,
            n_adjs / n_total,
            n_adv / n_total,
            n_pron / n_total,
            avg_word_len,
            sentence_len,
            lexical_diversity,
            stop_ratio,
            punct_ratio
        ])

    return np.array(feats, dtype=np.float32)