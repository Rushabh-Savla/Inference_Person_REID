from __future__ import annotations

import rebuild.live_state_invariant as base

from rebuild.batch_nvdcf import BatchNvDCF


# The live recorder stays unchanged; reconciliation is the same strict NvDCF
# pipeline used for recorded videos. This prevents the live path from silently
# switching to a non-NvDCF tracker/resolver.
base.BatchPipelineStateInvariantJointAttributes = BatchNvDCF
run_live_state = base.run_live_state

