ALPHA Curate
The ALPHA_curate strategy automatically selects log slices based on user-specified prompts using text-to-video similarity search powered by the Cosmos embedding model.
How It Works
The selection process uses text-based prompts to retrieve and score video slices via similarity search against a LanceDB index of Cosmos embeddings:
Load scenarios from the configured YAML file (each scenario contains one or more prompts)
Generate text embeddings using the Cosmos model for each prompt
Query LanceDB to find video slices with embeddings similar to the text prompt
Filter results based on similarity threshold and camera names
(Optional) Apply SQL filter to restrict results to specific slice IDs from BigQuery
Deduplicate to keep only the highest-scoring result per base slice
Output selections with negative similarity as the strategy score (lower is higher priority)
Selection Parameters
Parameters are configured in prompts.yaml (path set in ALPHACurateConfig.prompt_yaml_path). The YAML file contains a list of scenarios under the scenarios key. Each scenario defines:
name: Unique identifier for the scenario (e.g., "camera_obstruction", "peds_crossing_road"). Will be appended to "ALPHA_curate_" for the selection strategy name
prompts: List of prompt configurations for this scenario, where each prompt includes:
prompt: Natural language description of what you want to find (e.g., "person riding a bicycle", "highway lanes merging together")
camera_names: List of camera names to search (commonly ["camera_front_wide", "camera_front_narrow"])
similarity_threshold: Minimum similarity score (0.0-1.0) to include results. Start with 0.3 and adjust based on results:
Lower threshold (e.g., 0.25): More results, may include false positives
Higher threshold (e.g., 0.4): Fewer, more precise results
sql_filter (optional): SQL query to filter results by metadata; must return slice_id column, can optionally return slice_score for hybrid ranking
multiply_prompt_and_sql_scores (optional): Boolean to control whether SQL scores are multiplied with prompt scores (default: true)
To ensure that your parameters are effective, it’s highly recommended that you use https://to/ALPHA to determine an appropriate prompt and similarity threshold. An upper similarity threshold should be set in the left sidebar of the ALPHA app to ensure that the false positive rate is acceptable for your use case.
Example:
scenarios:
  - name: "peds_crossing_road"
    prompts:
      - prompt: "People running across the road"
        camera_names: ["camera_front_wide", "camera_front_narrow"]
        similarity_threshold: 0.3
        sql_filter: |
          SELECT slice_id, importance_score as slice_score
          FROM `project.dataset.metadata`
          WHERE city = 'Detroit'
Similarity scores are computed by converting L2 distances to a 0-1 range using the formula: similarity = max(0, 1 - distance/2).
Adding a New Scenario
To add a new selection scenario to ALPHA Curate, simply edit resources/prompts.yaml and add a new entry to the scenarios list:
scenarios:
  # ... existing scenarios ...

  - name: "my_new_scenario"
    prompts:
      - prompt: "Description of the driving scenario you want to find"
        camera_names: ["camera_front_wide", "camera_front_narrow"]
        similarity_threshold: 0.3
      - prompt: "Another related scenario description"
        camera_names: ["camera_front_wide"]
        similarity_threshold: 0.35
        # Optional SQL filtering
        sql_filter: |
          SELECT slice_id
          FROM `project.dataset.log_metadata`
          WHERE city = 'Detroit'
        # Optional: disable score multiplication (default is true)
        multiply_prompt_and_sql_scores: false
See the Selection Parameters section above for detailed parameter descriptions and guidance on choosing appropriate values.
Run ALPHA Curate with your new scenario to verify it finds relevant slices. The scenario will automatically be processed when the strategy runs.
The selection can be run locally. First, ensure that you have valid AWS credentials and valid GCP credentials.
GCP credentials can be generated via
gcloud auth login --enable-gdrive-access --no-launch-browser --update-adc
Then run
bazel run //autonomy/perception/datasets/active_learning/ALPHA_curate:generate_ALPHA_curate
To test on dagster, ensure that you’ve incremented the stage version in stage.py and then run
bazel run platforms/dagster/tools/cloud:cli -- deploy -c platforms/dagster/tools/cloud/resources/codeloc_configs/sensing.yaml
Once the branch deployment is created, navigate to the Dagster Cloud UI to launch a new run of the
ALPHA_curate_selection
 job.
SQL Filtering
The selection process supports optional SQL-based filtering to narrow down results based on metadata or criteria stored in BigQuery. This is useful when you want to:
Select slices from specific geographic regions
Filter by time of day, weather conditions, or other metadata
Target slices with specific tags or annotations
Combine multiple criteria using SQL logic
To use SQL filtering, add a sql_filter field to your prompt configuration in prompts.yaml. The SQL query must return a column named slice_id containing the slice IDs you want to include in the search results. Optionally, you can also return a slice_score column to provide SQL-specific scores for each slice.
Example SQL query (filtering only):
SELECT slice_id
FROM `project.dataset.log_metadata`
WHERE city = 'Detroit'
  AND weather_condition = 'rain'
  AND time_of_day BETWEEN '18:00:00' AND '22:00:00'
Example SQL query (with scoring):
SELECT slice_id, importance_score as slice_score
FROM `project.dataset.log_metadata`
WHERE city = 'Detroit'
Score Combination
When both a text prompt and SQL query with scores are provided:
The SQL filter is applied after the similarity search to restrict results to matching slice IDs
By default, if the SQL query returns a
slice_score column, the SQL scores are multiplied with the prompt similarity scores to produce a hybrid ranking
This allows you to combine semantic similarity (from the text prompt) with other relevance signals (from your SQL query)
To disable score multiplication, set multiply_prompt_and_sql_scores: false in the prompt configuration
A common use case is prioritizing one platform over another. For example, to prefer M-2 platform slices 4x more than M-1 platform slices:
SELECT slice_id,
  CASE
    WHEN platform_version_tag LIKE 'M-1%' THEN 1.0
    WHEN platform_version_tag LIKE 'M-2%' THEN 4.0
    ELSE 0.0
  END as slice_score
FROM `lat-bigquery-prod-a872.ontology.app_slices`
WHERE platform_version_tag LIKE 'M-1%' OR platform_version_tag LIKE 'M-2%'
The final score for each slice will be: final_score = prompt_similarity_score * sql_slice_score