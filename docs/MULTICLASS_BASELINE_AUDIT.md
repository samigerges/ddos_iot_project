# Paper baseline audit: confirmed protocol differences

This is the historical audit of the earlier raw-feature experiment. The later
[frozen final study](MULTICLASS_FINAL_RESULTS.md) completed the published-subset
reproduction: pinned-runtime test Macro-F1 is 0.890307 for eight classes and
0.887564 for thirty-four classes. The separate local protocol below remains
distinct from that published-data track.

The local eight-class reference (0.7904 validation Macro-F1) is **not an exact
reproduction** of the published approximately 0.8903 test result. This audit
identifies material differences; their individual score effects are unmeasured.

Sources: [Scientific Reports paper](https://www.nature.com/articles/s41598-026-47399-5),
[version-3 Zenodo record](https://zenodo.org/records/18675263), and
[pinned author source](https://github.com/sa1ah-ai/iot-intrusion-detection-gradient-boosting/tree/e4e0edbacd658721f28b657f16a7f68d6fa72073).
The concept DOI `10.5281/zenodo.17070208` resolves to this supplementary project;
version 3 has DOI `10.5281/zenodo.18675263`. Author scripts were parsed as syntax,
never imported/executed. Downloaded source URLs and SHA-256 values are retained.

## Differences to reconcile

| Component | Author version 3 | Local comparison |
|---|---|---|
| Cohort | Undersample original raw part files to about 5 million rows, then 80/20 split | All 5,491,971 rows of an existing training partition, then another internal 80/20 split |
| Minority retention | Minimum allocation of 1,000 per raw label in pinned code, before family mapping | Several raw classes have fewer than 1,000; Uploading_Attack has 140 |
| Eight-class XGBoost | Depth 10, fixed 500 rounds | Depth 6; reference refit 160 rounds after early stopping |
| Sampling/bins | Library defaults for row/column sampling and bins | Subsample 0.9, column sample 0.9, 128 bins |
| LightGBM selector | Unweighted; 100 rounds, 31 leaves, minimum leaf size 100 | Inverse-weighted; 50 estimators, 15 leaves, 63 bins |
| CV preprocessing | Select/scale once before CV | Select/scale inside each fold's inner fitting partition |
| 34-class XGBoost | Depth 14 for all features, depth 10 for selected features | Shared experiment configuration uses depth 6 |
| Versions | Paper reports Python 3.11.2, XGBoost 3.0.2 | Python 3.14.0, XGBoost 3.2.0 |

Family mapping and inverse-weighting agree in form; different class counts still
produce different weights. Equal feature counts do not establish equal subsets.
The paper describes family-level minimum allocation, but the released script
samples raw labels before mapping. Its XGBoost ablation uses fixed dictionaries;
it does not provide a reproduced grid-search trace. Prefer the released datasets
over assumptions about unspecified sampling behavior.

The author XGBoost script supplies test data as an evaluation set for logging,
but configures no XGBoost early stopping. This alone does not establish adaptive
test-based model selection.

## Reference data and next run

The versioned record lists `datasets.tar.xz`, **681,594,732 bytes**, with published
MD5 `2f21c060b062f95e7107fce24745124d`. It is kept in `data/raw/paper_v3/`, separate
from the existing train/test sources.

`src/prepare_paper_training.py` verifies the archive, computes SHA-256, and extracts
only the 8-class and 34-class training CSVs. It inventories labels and the 23
numeric columns without extracting/parsing test CSVs. Decompression traverses
compressed bytes containing other members; this is not test evaluation.

Preparation completed: **each training subset has 3,999,998 valid rows and 23
features**. All expected labels are present. The eight-class subset has 4,795
WebBased training rows, compared with 2,257 in our local development split.
Published and local selected-feature sets share 21 of their 23 columns. These
are concrete differences, not measured explanations for the score gap.

The verified archive SHA-256 is
`d84f77b115909d0feda42fc335af7f30274e647372c021c264380914f19c3e7f`.
No paper fidelity model has been trained on these subsets yet.

Released features are already selected/scaled using the full author training
split. Internal CV on them cannot be presented as fold-local feature selection.
A true all-46-feature/nested reproduction requires original raw source data and
sampling lineage. The next fidelity run should use the released training subset,
pinned selected-feature settings, and a version-matched environment where
available. Keep that benchmark separate from the stricter local OOF protocol.
Freeze reproduction settings before a separately declared paper-test evaluation.

## Artifacts and commands

- [Generated protocol comparison](../outputs/baseline_audit_v1/REPORT.md)
- [Machine-readable audit](../outputs/baseline_audit_v1/audit.json)
- [Parsed author settings](../outputs/baseline_audit_v1/author_settings.json)
- [Pinned source manifest](../outputs/baseline_audit_v1/sources/author_manifest.json)
- [Version-specific file listing](../outputs/baseline_audit_v1/sources/zenodo_v3_record.json)
- [Archive download/checksum status](../outputs/baseline_audit_v1/download.json)
- [Verified training counts, features, and file hashes](../outputs/baseline_audit_v1/published_training_inventory.json)

```powershell
.\.venv\Scripts\python.exe -B src/audit_paper_baseline.py --run-dir outputs/experiments/8class_v1
.\.venv\Scripts\python.exe -B src/prepare_paper_training.py
```
