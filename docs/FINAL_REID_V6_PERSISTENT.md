# Final V6 Persistent ReID

This branch is the finalization candidate built from `reference/v6-known-good` at `13be637928cd286ced2ed6dfb12b9fffd705e35c`.

The V6 ReID model, preprocessing, detector/tracker path, body-only gallery logic and thresholds are intentionally unchanged. The only behavioral addition is a durable live identity lifecycle.

## Permanent identity lifecycle

The live identity system now has two layers:

- `ActiveIdentitySet`: bounded in-memory execution state used by the proven V6 matcher.
- `PersistentIdentityRegistry`: SQLite-backed permanent GID namespace and exemplar gallery.

An active identity may be evicted from memory by the normal sweep. It is never deleted from the permanent registry by a sweep. When a new track reaches the V6 evidence gate, the persistent gallery is hydrated before matching.

The allocator is transactional and monotonic. A process restart, tracker reset, empty scene, or camera reconnect cannot reset the namespace to `G000001`.

## First deployment

Start with a new persistent state database so old experimental identities do not contaminate the final system:

```bash
rm -f identity_state/reid_live_v6.sqlite3 identity_state/reid_live_v6.sqlite3-shm identity_state/reid_live_v6.sqlite3-wal
```

Then use the default path or explicitly set:

```bash
export REID_IDENTITY_DB="$PWD/identity_state/reid_live_v6.sqlite3"
export REID_MODEL_ID="v6-live"
```

Do not delete this database between ordinary runs. It is the permanent identity memory.

## Required lifecycle proof

The final live system must demonstrate:

```text
A -> G000001
A leaves
scene empty
B -> G000002
A returns -> G000001
```

and, after a process restart:

```text
existing GIDs remain
next newly allocated GID continues monotonically
```

## Important limitation

This fixes the identity reset/lifecycle architecture. It does not make an imperfect ReID model mathematically perfect. The final A6000 run is still required to validate the real cameras and confirm that the known-good V6 identity behavior is preserved in the deployed environment.
