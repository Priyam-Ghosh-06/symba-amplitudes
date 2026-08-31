"""Baselines every headline number must be reported against (01 SS6.3).

The corpus is 30 (QED) and 11 (QCD) distinct functions instantiated with
different flavour labels, so a nearest-neighbour retriever does well without
learning anything. Any architectural claim has to clear that bar by more than
the confidence interval, and there is no way to know whether it does unless the
bar is measured.
"""

from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .metrics import evaluate_predictions


def _input_text(record) -> str:
    return " ".join(record["graph_tokens"]) + " || " + " ".join(record["amp_tokens"])


def most_frequent(train, test):
    """Always answer the most common training target."""
    counts = Counter(tuple(r["target_tokens"]) for r in train)
    answer = list(counts.most_common(1)[0][0])
    return [answer for _ in test]


def exact_lookup(train, test):
    """Answer from an exact match on the input, else the most frequent target."""
    table = {}
    for record in train:
        table.setdefault(_input_text(record), record["target_tokens"])
    fallback = most_frequent(train, [None])[0]
    return [table.get(_input_text(r), fallback) for r in test]


def nearest_neighbour(train, test, ngram_range=(3, 5)):
    """1-NN over character n-gram TF-IDF of the input. No training."""
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range)
    train_matrix = vectorizer.fit_transform([_input_text(r) for r in train])
    test_matrix = vectorizer.transform([_input_text(r) for r in test])

    similarity = cosine_similarity(test_matrix, train_matrix)
    best = similarity.argmax(axis=1)
    return [train[int(i)]["target_tokens"] for i in best]


def template_oracle(train, test):
    """Upper bound if the template class were known and only binding remained.

    Answers with a training record from the same template class, so it is
    correct whenever a class member happens to be an exact match and wrong
    otherwise. This is the ceiling pure retrieval can reach on this corpus.
    """
    by_template = {}
    for record in train:
        by_template.setdefault(record["template"], []).append(record)

    fallback = most_frequent(train, [None])[0]
    predictions = []
    for record in test:
        members = by_template.get(record["template"])
        if not members:
            predictions.append(fallback)
            continue
        exact = next((m for m in members
                      if m["target_tokens"] == record["target_tokens"]), None)
        predictions.append((exact or members[0])["target_tokens"])
    return predictions


BASELINES = {
    "most_frequent": most_frequent,
    "exact_lookup": exact_lookup,
    "nearest_neighbour": nearest_neighbour,
    "template_oracle": template_oracle,
}


def run_all(train, test, include_oracle: bool = True) -> dict:
    references = [r["target_tokens"] for r in test]
    templates = [r["template"] for r in test]

    results = {}
    for name, fn in BASELINES.items():
        if name == "template_oracle" and not include_oracle:
            continue
        predictions = fn(train, test)
        results[name] = evaluate_predictions(predictions, references, templates)
    return results
