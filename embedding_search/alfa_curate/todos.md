### Proposed refinement tasks (ranked by importance)
- 1) [x] Fix and decouple `alfa_curate/utils.py` (critical)
  - Implement headless helpers with proper imports and caching: `get_lakefs`, `load_table`, `load_model`, `text_to_embedding`.
  - Define `REPO`/`TABLE` correctly.
  - Update `alfa_curate/slice_scorer.py` to import from `alfa_curate/utils.py` (not the Streamlit app).

- 2) [x] Add per‑video dedup before scoring
  - Parse base video id from `row_id` and keep only the best hit per video.

- 3) [ ] Add dynamic threshold adjustment and noise prevention
  - Implement adaptive thresholding based on the quality of results to prevent selecting noise when good matches are exhausted
  - Add minimum similarity bounds that increase when fewer high-quality matches are found
  - Consider implementing a decay factor for prompts that consistently return low-quality results

- 4) [ ] Improve error handling and operational robustness
  - Add proper exception handling for LanceDB connection failures with retry logic
  - Validate YAML prompt configuration at startup (ensure prompts are non-empty, thresholds are in valid range)
  - Add logging for debugging which prompts are contributing to selections and their effectiveness
  - Handle edge cases like empty candidate sets or failed embedding generation gracefully

- 5) [ ] Optimize performance for large-scale processing
  - Batch embedding generation for multiple prompts to reduce model loading overhead
  - Consider caching query results for repeated prompts within a scoring session
  - Add configurable rate limiting to control the selection rate as mentioned in requirements
  - Implement parallel query execution for multiple prompts when the candidate set is large