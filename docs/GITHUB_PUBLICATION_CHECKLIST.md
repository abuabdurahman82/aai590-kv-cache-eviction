# GitHub Publication Checklist

## Before Upload

- [ ] Retain the repository as **Public** only if the course and team agree that the report and code can be shared publicly.
- [ ] Use this package as the repository root; do not upload its containing folder.
- [ ] Confirm no Google Drive paths, access tokens, personal identifiers beyond approved team names, or downloaded model files remain in notebook outputs.
- [ ] Keep only the canonical notebooks under `notebooks/`; avoid duplicate `v1`, `v2`, or temporary patch notebooks.
- [ ] Keep `kvcore_bundle.zip` and the visible `kvcore/` source together so the Colab workflow is inspectable and runnable.

## Upload Order in the GitHub Web Interface

1. Open the repository and select **Add file → Upload files**.
2. Upload the folders and root files from this package in a single batch if possible: `notebooks`, `kvcore`, `docs`, `results`, `artifacts`, `scripts`, `.github`, `README.md`, `requirements.txt`, `kvcore_bundle.zip`, and `.gitignore`.
3. Use the commit message: `Publish complete reproducible capstone workflow`.
4. After upload, open `README.md` and confirm that all folder links render and the notebook execution-order table is visible.
5. Open `notebooks/00_foundations_environment_eda.ipynb` and `notebooks/04_a100_locked_test_serving.ipynb` in the GitHub viewer to confirm the source is readable.
6. Confirm that GitHub’s file list does **not** show `*.pt`, `*.jsonl`, trace/cache folders, or raw Drive directories.

## Git Command Alternative

```bash
git clone https://github.com/abuabdurahman82/aai590-kv-cache-eviction.git
cd aai590-kv-cache-eviction
# Copy the contents of this package into the cloned directory.
git add .
git status
git commit -m "Publish complete reproducible capstone workflow"
git push origin main
```

## Final Grading Demonstration

Point the grader to these items first:

1. `README.md` for the scientific scope and evidence boundary.
2. `notebooks/00` through `notebooks/05` for the complete runnable workflow.
3. `kvcore/` for shared functional code and tests.
4. `scripts/validate_colab_v2.py` for suite validation.
5. `results/A100_LOCKED_TEST_EVIDENCE_AUDIT.md` for locked-test provenance and responsible result interpretation.
