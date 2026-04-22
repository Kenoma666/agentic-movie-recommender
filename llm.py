"""
Movie recommendation helpers.

The grader injects OLLAMA_API_KEY at runtime. This implementation can still
produce a deterministic recommendation when the key is absent or the API call
fails, but it will use the configured Ollama model when available.
"""

import argparse
import json
import math
import os
import re
import time
from collections import Counter

import ollama
import pandas as pd

MODEL = "gemma4:31b-cloud"
DATA_PATH = os.path.join(os.path.dirname(__file__), "tmdb_top1000_movies.csv")
MAX_DESCRIPTION_CHARS = 500
LLM_SHORTLIST_SIZE = 12

SEARCH_FIELD_WEIGHTS = {
    "title": 6.0,
    "genres": 5.5,
    "keywords": 4.5,
    "overview": 2.5,
    "tagline": 1.8,
    "top_cast": 1.2,
    "director": 1.2,
}

STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "for",
    "from",
    "good",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "me",
    "movie",
    "movies",
    "my",
    "of",
    "on",
    "or",
    "feel",
    "show",
    "something",
    "someth",
    "that",
    "the",
    "their",
    "them",
    "these",
    "this",
    "to",
    "want",
    "with",
}

PHRASE_EXPANSIONS = {
    "superhero": ["superhero", "comic", "vigilante", "powers", "hero"],
    "superheroes": ["superhero", "comic", "vigilante", "powers", "hero"],
    "feel good": ["heartwarming", "uplifting", "charming", "sweet", "hopeful", "family", "friendship"],
    "feel-good": ["heartwarming", "uplifting", "charming", "sweet", "hopeful", "family", "friendship"],
    "science fiction": ["science", "fiction", "scifi", "space", "future", "alien"],
    "sci fi": ["science", "fiction", "scifi", "space", "future", "alien"],
    "sci-fi": ["science", "fiction", "scifi", "space", "future", "alien"],
    "romantic comedy": ["romance", "romantic", "comedy", "funny", "charming"],
    "time travel": ["time", "travel", "timeline", "future", "past"],
    "coming of age": ["coming", "age", "teen", "growing", "youth"],
}

TOKEN_EXPANSIONS = {
    "action": ["thriller", "adventure", "combat", "chase", "explosive"],
    "adventure": ["quest", "journey", "epic"],
    "alien": ["space", "science", "fiction"],
    "animated": ["animation", "family"],
    "animation": ["animated", "family"],
    "comedy": ["funny", "humor", "laugh", "witty"],
    "crime": ["heist", "gangster", "detective"],
    "dark": ["gritty", "intense"],
    "detective": ["mystery", "investigation", "crime"],
    "family": ["heartwarming", "uplifting"],
    "fantasy": ["magic", "myth", "legend"],
    "funny": ["comedy", "humor", "laugh", "witty"],
    "heist": ["robbery", "crime", "thriller"],
    "horror": ["scary", "terror", "monster", "supernatural"],
    "light": ["uplifting", "charming", "sweet"],
    "mystery": ["detective", "investigation", "thriller"],
    "romance": ["romantic", "love", "relationship"],
    "romantic": ["romance", "love", "relationship"],
    "scary": ["horror", "terror", "supernatural"],
    "superhero": ["comic", "hero", "vigilante", "powers"],
    "thriller": ["suspense", "mystery", "crime", "intense"],
}

TEXT_NORMALIZATIONS = {
    "sci-fi": "science fiction",
    "scifi": "science fiction",
    "sci fi": "science fiction",
    "feel-good": "feel good",
    "superheroes": "superhero",
    "rom-com": "romantic comedy",
}

TOKEN_PATTERN = re.compile(r"[a-z0-9']+")

try:
    TOP_MOVIES = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    TOP_MOVIES = pd.read_excel(DATA_PATH.replace(".csv", ".xlsx"))

TOP_MOVIES.columns = [c.lower() for c in TOP_MOVIES.columns]
ID_COL = "id" if "id" in TOP_MOVIES.columns else "tmdb_id"
if ID_COL != "tmdb_id":
    TOP_MOVIES = TOP_MOVIES.rename(columns={ID_COL: "tmdb_id"})
    ID_COL = "tmdb_id"


def _safe_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def _normalize_text(text: str) -> str:
    normalized = _safe_text(text).lower()
    for src, dst in TEXT_NORMALIZATIONS.items():
        normalized = normalized.replace(src, dst)
    return normalized


def _stem_token(token: str) -> str:
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) > 5:
        token = token[:-3]
    elif token.endswith("ed") and len(token) > 4:
        token = token[:-2]
    elif token.endswith("es") and len(token) > 4:
        token = token[:-2]
    elif token.endswith("s") and len(token) > 3 and not token.endswith("ss"):
        token = token[:-1]
    return token


def _tokenize(text: str) -> list[str]:
    tokens = []
    for raw_token in TOKEN_PATTERN.findall(_normalize_text(text)):
        if raw_token in STOPWORDS:
            continue
        token = _stem_token(raw_token)
        if len(token) < 2 or token in STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def _build_query_counter(preferences: str) -> tuple[str, Counter, list[str]]:
    normalized_preferences = _normalize_text(preferences)
    query_counter = Counter(_tokenize(normalized_preferences))
    matched_phrases = []

    for phrase, expansions in PHRASE_EXPANSIONS.items():
        if phrase in normalized_preferences:
            matched_phrases.append(phrase)
            query_counter.update(_tokenize(" ".join(expansions)))

    for token in list(query_counter):
        expansions = TOKEN_EXPANSIONS.get(token, [])
        if expansions:
            query_counter.update(_tokenize(" ".join(expansions)))

    return normalized_preferences, query_counter, matched_phrases


def _prepare_search_index() -> tuple[dict[str, list[Counter]], list[str], dict[str, float], float]:
    field_counters = {}
    combined_text = []
    document_frequency = Counter()
    movie_count = len(TOP_MOVIES)

    for field in SEARCH_FIELD_WEIGHTS:
        series = TOP_MOVIES[field].fillna("") if field in TOP_MOVIES.columns else pd.Series([""] * movie_count)
        field_counters[field] = [Counter(_tokenize(value)) for value in series]

    for idx in range(movie_count):
        token_set = set()
        parts = []
        for field in SEARCH_FIELD_WEIGHTS:
            counter = field_counters[field][idx]
            token_set.update(counter.keys())
            parts.append(_normalize_text(_safe_text(TOP_MOVIES.at[idx, field])) if field in TOP_MOVIES.columns else "")
        document_frequency.update(token_set)
        combined_text.append(" ".join(part for part in parts if part))

    idf = {
        token: math.log((movie_count + 1) / (freq + 1)) + 1.0
        for token, freq in document_frequency.items()
    }
    default_idf = math.log(movie_count + 1) + 1.0

    return field_counters, combined_text, idf, default_idf


FIELD_COUNTERS, SEARCH_TEXT, TOKEN_IDF, DEFAULT_IDF = _prepare_search_index()


def _metadata_boost(row: pd.Series) -> float:
    vote_average = float(row.get("vote_average") or 0.0)
    vote_count = float(row.get("vote_count") or 0.0)
    popularity = float(row.get("popularity") or 0.0)
    return (
        0.18 * vote_average
        + 0.06 * math.log1p(max(vote_count, 0.0))
        + 0.03 * math.log1p(max(popularity, 0.0))
    )


def _history_title_penalty(candidate_idx: int, history: list[str]) -> float:
    if not history:
        return 0.0

    candidate_tokens = set(FIELD_COUNTERS["title"][candidate_idx])
    if not candidate_tokens:
        return 0.0

    penalty = 0.0
    for title in history:
        history_tokens = set(_tokenize(title))
        if not history_tokens:
            continue
        overlap = len(candidate_tokens & history_tokens)
        if overlap >= max(1, min(len(candidate_tokens), len(history_tokens))):
            penalty += 4.0
    return penalty


def _score_candidate(
    idx: int,
    query_counter: Counter,
    matched_phrases: list[str],
    history: list[str],
) -> float:
    score = 0.0
    for token, query_weight in query_counter.items():
        token_idf = TOKEN_IDF.get(token, DEFAULT_IDF)
        for field, field_weight in SEARCH_FIELD_WEIGHTS.items():
            tf = FIELD_COUNTERS[field][idx].get(token, 0)
            if tf:
                score += field_weight * token_idf * query_weight * (1.0 + 0.2 * min(tf - 1, 2))

    searchable_text = SEARCH_TEXT[idx]
    for phrase in matched_phrases:
        if phrase in searchable_text:
            score += 6.0

    score += _metadata_boost(TOP_MOVIES.iloc[idx])
    score -= _history_title_penalty(idx, history)
    return score


def _rank_candidates(
    preferences: str,
    history: list[str],
    history_ids: list[int],
) -> tuple[pd.DataFrame, Counter, list[str]]:
    _, query_counter, matched_phrases = _build_query_counter(preferences)

    if not query_counter:
        query_counter.update(_tokenize(preferences))

    candidates = TOP_MOVIES[~TOP_MOVIES[ID_COL].isin(history_ids)].copy()
    candidate_scores = [
        _score_candidate(int(idx), query_counter, matched_phrases, history)
        for idx in candidates.index
    ]
    candidates["_retrieval_score"] = candidate_scores
    candidates = candidates.sort_values(
        by=["_retrieval_score", "vote_average", "vote_count", "popularity"],
        ascending=[False, False, False, False],
    )
    return candidates, query_counter, matched_phrases


def _compact_text(value, limit: int) -> str:
    text = " ".join(_safe_text(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def build_prompt(
    preferences: str,
    history: list[str],
    history_ids: list[int],
    shortlist: pd.DataFrame,
) -> str:
    history_text = (
        ", ".join(f'"{name}" (tmdb_id={tid})' for name, tid in zip(history, history_ids))
        if history
        else "none"
    )

    candidate_lines = []
    for rank, (_, row) in enumerate(shortlist.iterrows(), start=1):
        candidate_lines.append(
            (
                f"{rank}. tmdb_id={int(row[ID_COL])} | title={_safe_text(row.get('title', ''))} | "
                f"genres={_compact_text(row.get('genres', ''), 80)} | "
                f"keywords={_compact_text(row.get('keywords', ''), 120)} | "
                f"overview={_compact_text(row.get('overview', ''), 220)} | "
                f"director={_compact_text(row.get('director', ''), 60)}"
            )
        )

    candidate_block = "\n".join(candidate_lines)

    return f"""You are a precise movie recommendation assistant.

Choose exactly ONE movie from the ranked shortlist below. The shortlist was
already pre-filtered for relevance, so stay inside it and prefer the strongest
match in tone, genre, themes, and keywords.

User preferences:
{preferences}

Watch history (do not recommend these):
{history_text}

Ranked shortlist:
{candidate_block}

Return ONLY valid JSON with this exact schema:
{{
  "tmdb_id": <integer from the shortlist>,
  "description": "<short pitch under 300 characters explaining why the movie fits>"
}}"""


def call_llm(prompt: str, api_key: str | None) -> dict | None:
    if not api_key:
        return None

    try:
        client = ollama.Client(
            host="https://ollama.com",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        response = client.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0.2},
        )
        content = response.message.content.strip()
        if content.startswith("```json"):
            content = content[7:-3]
        elif content.startswith("```"):
            content = content[3:-3]
        return json.loads(content)
    except Exception:
        return None


def _pick_reason_tokens(row: pd.Series, query_counter: Counter) -> list[str]:
    reason_tokens = []
    for field in ("genres", "keywords", "overview", "title"):
        movie_tokens = set(FIELD_COUNTERS[field][int(row.name)])
        for token in query_counter:
            if token in movie_tokens and token not in reason_tokens:
                reason_tokens.append(token)
            if len(reason_tokens) >= 3:
                return reason_tokens
    return reason_tokens


def _build_local_description(row: pd.Series, query_counter: Counter) -> str:
    title = _safe_text(row.get("title", "This movie"))
    genres = _compact_text(row.get("genres", ""), 60)
    reason_tokens = _pick_reason_tokens(row, query_counter)

    if reason_tokens:
        token_text = ", ".join(reason_tokens[:3])
        description = (
            f"{title} stands out for {token_text}, with {genres.lower()} elements "
            f"that line up well with what you asked for."
        )
    elif genres:
        description = f"{title} is a strong fit thanks to its {genres.lower()} mix and broad audience appeal."
    else:
        description = f"{title} is a strong fit based on your preferences and the movie metadata."

    return description[:MAX_DESCRIPTION_CHARS]


def _validate_llm_choice(shortlist: pd.DataFrame, parsed: dict | None) -> tuple[int | None, str]:
    if not parsed:
        return None, ""

    try:
        tmdb_id = int(parsed.get("tmdb_id"))
    except Exception:
        return None, ""

    valid_ids = set(shortlist[ID_COL].astype(int))
    if tmdb_id not in valid_ids:
        return None, ""

    description = " ".join(_safe_text(parsed.get("description", "")).split())
    return tmdb_id, description[:MAX_DESCRIPTION_CHARS]


def get_recommendation(
    preferences: str,
    history: list[str],
    history_ids: list[int] | None = None,
) -> dict:
    """Return a dict with keys 'tmdb_id' and 'description'."""

    history_ids = history_ids or []
    ranked_candidates, query_counter, _ = _rank_candidates(preferences, history, history_ids)
    shortlist = ranked_candidates.head(LLM_SHORTLIST_SIZE)
    best_row = shortlist.iloc[0]

    prompt = build_prompt(preferences, history, history_ids, shortlist)
    parsed = call_llm(prompt, os.environ.get("OLLAMA_API_KEY"))
    llm_tmdb_id, llm_description = _validate_llm_choice(shortlist, parsed)

    if llm_tmdb_id is not None:
        chosen_row = shortlist[shortlist[ID_COL].astype(int) == llm_tmdb_id].iloc[0]
        description = llm_description or _build_local_description(chosen_row, query_counter)
        return {"tmdb_id": int(llm_tmdb_id), "description": description}

    return {
        "tmdb_id": int(best_row[ID_COL]),
        "description": _build_local_description(best_row, query_counter),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a local movie recommendation test.")
    parser.add_argument("--preferences", type=str, help="User preferences text.")
    parser.add_argument("--history", type=str, help="Comma-separated watch history titles.")
    args = parser.parse_args()

    print("Movie recommender - type your preferences and press Enter.")
    preferences = args.preferences.strip() if args.preferences else input("Preferences: ").strip()
    history_raw = args.history.strip() if args.history else input("Watch history (optional): ").strip()
    history = [title.strip() for title in history_raw.split(",") if title.strip()] if history_raw else []

    print("\nThinking...\n")
    start = time.perf_counter()
    result = get_recommendation(preferences, history)
    movie_match = TOP_MOVIES[TOP_MOVIES[ID_COL].astype(int) == int(result["tmdb_id"])]
    if not movie_match.empty:
        cli_result = {
            "tmdb_id": int(result["tmdb_id"]),
            "title": _safe_text(movie_match.iloc[0].get("title", "")),
            "description": result["description"],
        }
    else:
        cli_result = result
    print(json.dumps(cli_result, indent=2, ensure_ascii=False))
    elapsed = time.perf_counter() - start

    print(f"\nServed in {elapsed:.2f}s")
