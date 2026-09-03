from __future__ import annotations

import rebuild.live_state_invariant as base

from rebuild.batch_state_invariant_accurate import BatchPipelineStateInvariantAccurate


# The capture implementation stays unchanged. Only the Safe055/V6 video
# reconciliation class is replaced with the strict feature-first implementation.
base.BatchPipelineStateInvariantJointAttributes = BatchPipelineStateInvariantAccurate
run_live_state = base.run_live_state
