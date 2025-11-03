"""Asset definition for QwenVL video judgment features."""

from typing import Optional

import dagster

from autonomy.perception.datasets.features.qwen_vl.config import QwenVLJudgeConfig
from autonomy.perception.datasets.features.qwen_vl.stage import STAGE_DEFINITION
from kits.scalex.dataset.bluemind.scalex_launch_v2 import maybe_build_image_and_launch_job
from kits.scalex.dataset.manifest import Manifest
from kits.scalex.dataset.scalex_stage_v2 import scalex_stage_v2
from platforms.dagster.libs.artifactory import ArtifactoryResource
from platforms.dagster.libs.bluemind import BlueMindResource
from platforms.dagster.libs.jenkins import JenkinsResource
from platforms.dagster.libs.lakefs import LakeFSResource


@scalex_stage_v2(stage_def=STAGE_DEFINITION)
def features_qwenvl_judge(
    context: dagster.OpExecutionContext,
    log_slices_silver: Optional[Manifest],
    config: QwenVLJudgeConfig,
    *,
    lakefs_resource: LakeFSResource,
    bluemind: BlueMindResource,
    jenkins: JenkinsResource,
    artifactory: ArtifactoryResource,
) -> Manifest:
    """Create the QwenVL video judgment dataset."""
    return maybe_build_image_and_launch_job(
        context,
        lakefs_resource,
        bluemind,
        jenkins,
        artifactory,
        stage_def=STAGE_DEFINITION,
        config=config,
        log_slices_silver_reference=log_slices_silver,
    )

