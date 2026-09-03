from __future__ import annotations

import rebuild.live_state_invariant as base

from rebuild.batch_state_invariant_overlap_reid import BatchPipelineStateInvariantOverlapReid


# Keep concurrent FFmpeg capture unchanged. Swap only the reconciliation class
# so recorded live captures use the same hard-overlap pause/recovery path.
base.BatchPipelineStateInvariantJointAttributes = BatchPipelineStateInvariantOverlapReid
run_live_state = base.run_live_state
