I think that there should hopefully be a straightforward path for you to land this relatively quickly:


Create a YAML config file with a list of prompts.  Each prompt may need some additional metadata, like a threshold.


Implement an active learning strategy within the framework that Subclasses SliceScorerBase.  The setup here is different than most other selection strategies:
process_slice should just return the slice id
compute_slice_scores should aggregate slice ids, run one query for each prompt, and then filter the results to just those slice ids that are in the set of slice ids to consider
Longer term SML may need to create a new base class for this type of selection strategy because it doesn't really fit the map/reduce pattern

We should be able to turn this on, especially if we set a small rate to start with.  Main thing will be to ensure that we set reasonable bounds/thresholds to prevent selecting noise if we've exhausted the slices that match a particular query.

template reference
/Users/lilyzhang/Desktop/Qwen2.5-VL/embedding_search/dinov2/config.py
/Users/lilyzhang/Desktop/Qwen2.5-VL/embedding_search/dinov2/dino_obstruction.py
- may not need @ray.remote class RemoteIndex:

refer to streamlit_app_v2.py on how the cosmos embedding get used for text search
/Users/lilyzhang/Desktop/Qwen2.5-VL/embedding_search/interface/streamlit_app_v2.py