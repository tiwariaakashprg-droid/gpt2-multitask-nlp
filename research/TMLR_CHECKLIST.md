# TMLR Submission Preparation Checklist

## Experimental changes implemented in this package

- [x] Explicit judge scoring rubric added.
- [x] Multi-judge runner added.
- [x] Human-human pairwise agreement available from anonymized ratings.
- [x] ICC(2,k) analysis added.
- [x] Bootstrap 95% confidence intervals added.
- [x] Partial correlation for length analysis added.
- [x] Judge-vs-judge comparison supported.
- [x] Ethics verification checklist added.
- [x] Annotator identifiers anonymized in the release package.
- [x] Reproducibility instructions added.

## Still required before claiming the strengthened experiment

- [ ] Re-run LLaMA-3 with the new v2 rubric.
- [ ] Run a second local judge with the same v2 rubric.
- [ ] Record exact judge model identifiers/digests.
- [ ] Verify human annotation consent/ethics facts.
- [ ] Decide whether additional questions/annotators are feasible.
- [ ] Update the manuscript from the new generated results only.
- [ ] Ensure the final public repository is anonymized for review.
- [ ] Recheck every numerical claim against generated output files.
