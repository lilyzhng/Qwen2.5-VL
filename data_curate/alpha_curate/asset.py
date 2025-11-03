"""Generate the alfa curate dataset."""


from typing import Optional

import dagster

from autonomy.perception.datasets.active_learning.alfa_curate.config import AlfaCurateConfig
from autonomy.perception.datasets.active_learning.alfa_curate.stage import STAGE_DEFINITION
from kits.scalex.dataset.bluemind.scalex_launch_v2 import maybe_build_image_and_launch_job
from kits.scalex.dataset.manifest import Manifest
from kits.scalex.dataset.scalex_stage_v2 import scalex_stage_v2
from platforms.dagster.libs.artifactory import ArtifactoryResource
from platforms.dagster.libs.bluemind import BlueMindResource
from platforms.dagster.libs.jenkins import JenkinsResource
from platforms.dagster.libs.lakefs import LakeFSResource


@scalex_stage_v2(stage_def=STAGE_DEFINITION)
def alfa_curate(
    context: dagster.OpExecutionContext,
    human_labels_gold_main: Optional[Manifest],
    log_slices_silver: Optional[Manifest],
    active_learning_filter: Optional[Manifest],
    config: AlfaCurateConfig,
    *,
    lakefs_resource: LakeFSResource,
    bluemind: BlueMindResource,
    jenkins: JenkinsResource,
    artifactory: ArtifactoryResource,
) -> Manifest:
    """Build the image and launch the job."""
    return maybe_build_image_and_launch_job(
        context,
        lakefs_resource,
        bluemind,
        jenkins,
        artifactory,
        stage_def=STAGE_DEFINITION,
        config=config,
        human_labels_gold_reference=human_labels_gold_main,
        log_slices_silver_reference=log_slices_silver,
        active_learning_filter_reference=active_learning_filter,
    )

