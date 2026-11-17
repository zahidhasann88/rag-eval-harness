import re
import string

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION = str.maketrans("", "", string.punctuation)
_WORD = re.compile(r"[^\W_]+")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from further
    had has have having he her here hers herself him himself his how i if in into is it its itself
    just me more most my myself no nor not now of off on once only or other our ours ourselves out
    over own same she should so some such than that the their theirs them themselves then there
    these they this those through to too under until up very was we were what when where which
    while who whom why will with would you your yours yourself yourselves
    """.split()
)


def normalize_answer(text: str) -> str:
    """Lowercase, strip ASCII punctuation and English articles, and collapse whitespace."""
    lowered = text.lower().translate(_PUNCTUATION)
    return " ".join(_ARTICLES.sub(" ", lowered).split())


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def fold_plural(token: str) -> str:
    if len(token) <= 3:
        return token
    if token.endswith("ies"):
        return token[:-3] + "y"
    if token.endswith("sses"):
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def content_terms(text: str) -> list[str]:
    return [fold_plural(token) for token in tokenize(text) if token not in STOPWORDS]


def split_sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in _SENTENCE_BOUNDARY.split(text) if sentence.strip()]
