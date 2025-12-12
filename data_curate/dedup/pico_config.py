from kits.scalex.dataset.config import BaseStageConfigV2

_LOGGER: Final = logging.getLogger(__name__)


class HumanLabelsPicoConfig(BaseStageConfigV2):
    """Config for generating the human labels pico dataset stage."""

    #: A string reference to the human labels gold stage. This dataset is currently hard-coded to refer to the dataset
    #: being used by the front-bev-p758 model. If we decided to automate this dataset, this will be updated.
    # The current front bev dataset is human labels bundled @main_2025-08-14.0
    # This corresponds the following gold commit:
    human_labels_gold_reference: str = (
        "sensing--human-labels--gold/main@729800ca27bdc6dc0da9e904cbd055dba736e409da4bbccb84feda685a2de976"
    )

    features_dinov2_index_reference: str = "sensing--features--dinov2-index/main"

    branch: str = "pico"

    #: Process all the references.
    limit: int = 0

    #: Use maximum number of CPUs for prod run.
    use_max_cluster_cpus_for_prod: bool = True

    #: Whether to have incremental materialization.
    incremental: bool = True

    #: Whether to use ray or not for processing the slices. It is easier to debug locally when ray is disabled.
    disable_ray: bool = False

    #: Setting to indicate a local run.
    local_run: bool = False

    #: The stride to use when downsampling groups. A stride of 1 does no downsampling.
    stride: int = 1

    #: The number of slices that are processed in each ray dataset. The default value of 5000 was chosen because it was
    #: large, but not too large that the cluster runs out of memory on a 96 CPU node.
    num_rows_per_partition: int = 5000

    #: The number of frames per pico row.
    frames_per_row: int = 10

    #: The final number of pico rows per file.
    final_rows_per_file: int = 2000

    #: Whether to stop after processing one partition. This option will process only the first num_rows_per_partition
    #: and then stop. It's useful during testing to avoid processing the entire dataset.
    testing_stop_after_one_partition: bool = False

    #: The number of KMeans clusters to use. This setting is an initial cluster count for the deduplication process.
    #: Deduplication is done in parallel within each cluster.
    num_kmeans_clusters: int = 1024

    #: The threshold for embedding deduplication as an L2 distance. When negative, no deduplication is performed.
    embedding_dedupe_threshold: float = -1

    #: The number of iterations to run for KMeans clustering. The default for FAISS is 25.
    kmeans_iterations: int = 25

    #: An integer representing the number of nearest neighbors to consider when calculating average distances for density estimates.
    nearest_k_points: int = 10

    #: The maximum number of points to use for clustering and deduplication. This parameter is useful only in the
    #: context of testing to limit the number of points used. A value of 50,000 is reasonable.
    testing_limit_clustering_points: int = 0

    # If True, subsample pico rows based on diversity scores
    apply_temporal_subsampling: bool = False

    # If True, compute acceleration (change of change rates). If False, compute velocity.
    use_acceleration: bool = True

    # Threshold for temporal subsample by diversity.
    diversity_threshold: float = 0.15

    # Comparison window for velocity/acceleration in seconds.
    temporal_window_size_s: float = 15.0

    # If True, enable VRU (Vulnerable Road User) preservation during deduplication
    enable_object_preservation: bool = False

    preserved_classes: Optional[List[str]] = None
    # List of object class names to preserve (e.g., pedestrians, cyclists, motorcyclists)
    # Frames containing these classes will not be pruned during deduplication
    @validator("preserved_classes", pre=True, always=True)
    def set_default_preserved_classes(cls, value: Optional[List[str]]) -> List[str]:
        """Set default preserved classes when None is provided."""
        if value is not None:
            return value
        
        return [
            "ADULT",
            "CHILD",
            "CONSTRUCTIONWORKER",
            "FIRSTRESPONDER",
            "OFFICIAL_SIGNALER",
            "PEDESTRIAN",
            "UNOFFICIAL_SIGNALER",
            "VULNERABLE_ROAD_USER",
            "ACCELERATEDHUMAN",
        ]