# Movie Recommender

This project implements a movie recommendation agent in `llm.py`.
Given a user's free-text preferences and watch history, it returns a
dictionary with:

- `tmdb_id`: the recommended movie ID from `TOP_MOVIES`
- `description`: a short explanation of why the movie fits

The implementation keeps the required model fixed as `gemma4:31b-cloud`
when calling Ollama's cloud API.

## Approach

The recommender uses a two-stage pipeline designed to balance relevance,
speed, and robustness.

### 1. Local retrieval and ranking

Instead of sending the full dataset to the model or sampling random movies,
the code first ranks the candidate pool locally using metadata from the CSV:

- `title`
- `genres`
- `keywords`
- `overview`
- `tagline`
- `director`
- `top_cast`

The ranking pipeline does the following:

1. Normalizes the user query with lightweight text cleanup.
2. Expands important phrases such as `superhero`, `feel-good`, and `sci-fi`
   into related terms.
3. Tokenizes both the query and movie metadata.
4. Scores each movie using weighted token overlap plus a small metadata boost
   from popularity and ratings.
5. Excludes movies that appear in the user's watch history.

This stage improves recommendation quality because the LLM only sees a strong
shortlist instead of a random subset of movies.

### 2. LLM selection from a shortlist

After ranking, the top candidates are passed to the Ollama model in a prompt
that asks it to choose exactly one movie from the shortlist and return JSON.

The model is used for the final judgment step:

- compare tone and genre fit
- choose the strongest match from the shortlist
- generate a concise recommendation pitch

This keeps the prompt small enough to stay fast while still using the model's
language understanding.

### 3. Deterministic fallback

If the API key is missing, the cloud request fails, or the model returns
invalid output, the program falls back to the highest-ranked local candidate.

This fallback is important for two reasons:

- the function still returns a valid recommendation quickly
- the system avoids random behavior, which makes results more stable

## Evaluation Strategy

I used two layers of evaluation.

### 1. Required automated tests

The project includes `test.py`, which checks:

- the returned value is a `dict`
- both `tmdb_id` and `description` are present
- the ID belongs to the provided candidate set
- the recommendation does not repeat a watched movie
- the function returns within the 20-second limit
- imports used in `llm.py` are covered by `requirements.txt`

Command used:

```bash
OLLAMA_API_KEY=your_key_here python test.py
```

In the current development environment, I also verified the code path with a
placeholder key when a real key was not available:

```bash
$env:OLLAMA_API_KEY='dummy'; python test.py
```

### 2. Manual spot checks

I also tested prompts manually to inspect recommendation quality, such as:

- superhero / action queries
- funny and feel-good queries
- tragedy queries
- sci-fi thriller queries

The main things I looked for were:

- whether the selected movie matched the requested mood or genre
- whether watch-history filtering worked
- whether the output stayed short and readable
- whether the response remained fast

## Brief Guide To The Code

The main file is `llm.py`. The most important parts are:

- `MODEL`, `SEARCH_FIELD_WEIGHTS`, and related constants:
  define the fixed cloud model and retrieval behavior
- `_tokenize()`, `_build_query_counter()`:
  normalize and expand the user's text input
- `_prepare_search_index()`:
  builds a lightweight searchable representation of the dataset
- `_score_candidate()` and `_rank_candidates()`:
  score each movie and produce the shortlist
- `build_prompt()`:
  formats the shortlist and user context for the LLM
- `call_llm()`:
  calls Ollama cloud with `gemma4:31b-cloud`
- `get_recommendation()`:
  orchestrates the full flow and applies fallback behavior
- `__main__` block:
  supports local CLI testing and prints the movie title for convenience

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the recommender directly:

```bash
OLLAMA_API_KEY=your_key_here python llm.py --preferences "I want a funny, light, action-packed movie." --history "The Dark Knight Rises"
```

Or run interactively:

```bash
OLLAMA_API_KEY=your_key_here python llm.py
```

Run the test suite:

```bash
OLLAMA_API_KEY=your_key_here python test.py
```

## Notes

- The grader only requires `tmdb_id` and `description` in the function return
  value.
- The CLI output additionally shows the movie `title` to make manual testing
  easier.
- No API keys are hard-coded. `OLLAMA_API_KEY` is always read from the
  environment.
