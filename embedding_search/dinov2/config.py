"""Implementation of selection the slices scored by DINOv2 based camera obstruction."""

from autonomy.perception.datasets.active_learning.camera_obstruction_dinov2.base_config import (
    BaseStrategyConfig,  # pragma: ignore-dep
)


class StrategyConfig(BaseStrategyConfig):
    """Configuration for the selection strategy."""

    #: Only score slices with P(obstructed) >= this threshold.
    obstruction_prob_threshold: float = 0.01

    #: Number of voting neighbors.
    #: TODO SP-961: increase the num_voting_neighbors to a larger number when the latency issue is resolved.
    num_voting_neighbors: int = 3

    #: Number of neighbours we query from the FAISS index.
    num_faiss_query_neighbors: int = 100

    #: Minimum voters required to compute a score. Its value should less than or equal to num_voting_neighbors.
    min_voters: int = 3

    #: Maximum weight for distance weighting to cap the weight of the closest neighbours to avoid extreme values
    max_weight: int = 1000

    #: Whether to instantiate object allowing to access all the slices in the dataset
    instantiate_ray_datasets: bool = True

    #: Name of the camera whose images should be scored.
    target_camera_name: str = "camera_front_wide"

    #: How many concurrent FAISS queries
    index_actor_concurrency: int = 16
