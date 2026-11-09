# Recall-constrained eight-class results: seed 42

Two policies were completed on all six cached models. No classifiers were
retrained; no local or published test CSV was parsed/evaluated. Limits came from
reference OOF argmax predictions on the same training rows. Both searches used
the fixed grid `[-4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4]`, three
coordinate passes, and two starts: zero bias and the previous OOF bias. Candidate
selection used feasible OOF Macro-F1 and was frozen with every bias before
validation features/labels were loaded.

## Validation operating points

| Operating point | Macro-F1 | Benign FPR | WebBased recall | Recon recall | BruteForce recall |
|---|---:|---:|---:|---:|---:|
| Original weighted reference, argmax | 0.790436 | 7.65% | 85.46% | 81.49% | 77.27% |
| Protect WebBased + BruteForce: OOF choice | 0.790265 | 4.47% | 86.35% | 76.63% | 78.25% |
| Protect all three: OOF choice | 0.786953 | 6.01% | 86.70% | 81.79% | 75.65% |
| Protect all three: reference model's tuned point | 0.790448 | 7.65% | 85.46% | 81.49% | 77.27% |

The two-target policy selected the median-feature weighted model with new biases.
Benign false alarms fell from 1,981 to 1,158 (41.5% fewer), but Macro-F1 was
essentially unchanged and Recon recall fell by 4.85 percentage points.

The exploratory follow-up added Recon after that error analysis. It selected
the all-feature inverse-weight model on OOF. On validation, however, BruteForce
detection fell from 238/308 to 233/308: five additional missed attacks. This choice
**fails the strict validation recall requirement** and is not promoted to a final
model. The reference model's alternative passed all three checks, but its
Macro-F1 gain of +0.000013 is effectively zero.

There is no clear recall-preserving Macro-F1 gain from this search. The earlier
unconstrained gain to 0.8779 was a precision/recall trade, not a free increase in
attack detection. This finite search does not prove no better bias/model exists.

## Scope

- Validation had already been used in earlier ablations; the three-target policy
  followed its error analysis. These are exploratory results, not fresh test
  evidence. One seed and small minority supports limit conclusions.
- Feasibility precedes Macro-F1 in the search. Failure means no feasible point
  was found by this heuristic, not proof that a model cannot meet the limits.
- No new latency measurement was made for these operating points.
- Original models, reports and frozen selection are unchanged. The parent
  `evaluate-test` command still uses its earlier unconstrained selection; it
  does not automatically adopt a constrained child run.

The next experiment is the [paper fidelity baseline](MULTICLASS_BASELINE_AUDIT.md)
with the actual published data/settings. Confirm it before stacking or final
test evaluation.

## Evidence and commands

- [Two-target report](../outputs/experiments/8class_v1/recall_constraints_v1/REPORT.md)
- [Three-target report](../outputs/experiments/8class_v1/recall_constraints_all_targets_v1/REPORT.md)
- [Every class: two targets](../outputs/experiments/8class_v1/recall_constraints_v1/per_class_comparison.csv)
- [Every class: three targets](../outputs/experiments/8class_v1/recall_constraints_all_targets_v1/per_class_comparison.csv)

```powershell
.\.venv\Scripts\python.exe -B src/retune_multiclass_recall.py --run-dir outputs/experiments/8class_v1 --output-dir outputs/experiments/8class_v1/recall_constraints_all_targets_v1 --protect WebBased Recon BruteForce
.\.venv\Scripts\python.exe -B src/report_recall_tuning.py --run-dir outputs/experiments/8class_v1/recall_constraints_all_targets_v1
```

Existing tuning outputs are protected from overwrite. Choose a fresh directory
for another run; report regeneration reads only saved metrics.

Independent checks recomputed every feasible OOF result (seven across the two
policies) and all seven validation confusion matrices. The two exported research
bundles reproduced 300 saved validation predictions each. Original training
implementation fingerprints and the parent frozen selection remained unchanged.
The regression run passed 89 tests with one optional CUDA integration test
skipped; three additional archive tests passed. Updated export integration tests
also passed.
