from __future__ import annotations

import re
from collections import Counter


TOKEN_RE = re.compile(r"[a-zA-Z0-9_.+-]+")


def normalize(text: str) -> str:
    return " ".join(tokenize(text))


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def term_present(text: str, term: str) -> bool:
    return normalize(term) in normalize(text)


def overlap_score(query: str, text: str) -> float:
    query_counts = Counter(tokenize(query))
    text_counts = Counter(tokenize(text))
    if not query_counts or not text_counts:
        return 0.0
    overlap = sum(min(query_counts[token], text_counts[token]) for token in query_counts)
    return overlap / max(1, sum(query_counts.values()))
