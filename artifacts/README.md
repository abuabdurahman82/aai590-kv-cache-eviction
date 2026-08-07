# External Artifact Inventory

The full experiment artifacts are preserved outside GitHub because they are large, generated during runtime, or contain a user-specific Drive path. They are **not required to inspect the functional code** in this repository.

## Excluded Artifact Classes

| Artifact class | Why it is excluded |
|---|---|
| Frozen scorer checkpoint (`*.pt`) | Binary model artifact; keep in controlled project storage. |
| Raw JSONL progress journal | Contains 1,000 individual locked-test measurements and is larger than the approved summary set. |
| Per-record trace and cache files | Large, generated runtime outputs. |
| Downloaded model weights and Hugging Face cache | Third-party model data; obtain through the pinned configuration. |
| Google Drive directories and authentication artifacts | User-specific and private. |

## Verification Path

The repository includes the A100 evidence audit and compact summary tables in [`../results/`](../results). The A100 notebook verifies the selected checkpoint path, test partition, resume manifest, and journal fingerprint before running or resuming the locked test.
