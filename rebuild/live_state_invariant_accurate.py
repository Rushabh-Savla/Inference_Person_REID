from __future__ import annotations

import rebuild.live_state_invariant as base
from rebuild import batch_state_invariant as state_pipeline

from rebuild.batch_state_invariant_overlap_reid import BatchPipelineStateInvariantOverlapReid
from rebuild.multimodel_state_invariant_overlap_reid import OverlapReidResolver


# Keep concurrent FFmpeg capture unchanged. Use the same hard-overlap pause,
# dense post-overlap ReID recovery, and trajectory-aware resolver for the
# recorded live session.
state_pipeline.StateInvariantFinalResolver = OverlapReidResolver
base.BatchPipelineStateInvariantJointAttributes = BatchPipelineStateInvariantOverlapReid
run_live_state = base.run_live_state
