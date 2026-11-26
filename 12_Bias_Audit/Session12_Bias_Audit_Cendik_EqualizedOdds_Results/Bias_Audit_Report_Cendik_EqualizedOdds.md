# Bias Audit Report — Individual Equalized-Odds Variant
**Student:** Cendik Leonardo Mauricio Orcón

## 1. Setup
- **Dataset:** ProPublica COMPAS two-year recidivism dataset loaded through AIF360.
- **Protected attribute:** `race`
- **Privileged group:** Caucasian (privileged)
- **Unprivileged group:** African-American (unprivileged)
- **Favorable outcome:** did NOT re-offend within two years.
- **Favorable label code:** `0.0`
- **Baseline:** Logistic Regression.
- **Individual mitigation:** ExponentiatedGradient + EqualizedOdds.
- **Equalized-Odds optimization bound:** `0.02`.
- **Scientific difference from the exemplar:** the exemplar prioritizes statistical parity with Reweighing; this submission prioritizes error-rate balance.

## 2. Bias already in the labels
- **Disparate impact in training labels:** `0.850`
- **Statistical parity difference in training labels:** `-0.090`

Historical group disparity exists before model training. This is important context, but my mitigation objective is not statistical parity; it is Equalized Odds.

## 3. BEFORE metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.664 |
| Disparate impact | 0.773 |
| Statistical parity difference | -0.165 |
| TPR difference / equal opportunity difference | -0.095 |
| FPR difference | -0.183 |
| Equalized-Odds gap | 0.183 |
| Average odds difference | -0.139 |

The primary baseline concern for this individual audit is the combination of TPR and FPR differences, summarized by the Equalized-Odds gap.

## 4. Mitigation applied and why
I use **ExponentiatedGradient with an EqualizedOdds constraint**, an in-processing mitigation. I chose this objective because a recidivism-risk model can harm groups through unequal false alarms as well as unequal misses. Equalized Odds therefore aligns more directly with the distribution of classification errors than statistical parity alone.

Fairlearn expects binary classification with `1` as the positive class, whereas AIF360 COMPAS defines the favorable class as `0`. For the mitigation only, I recoded `favorable -> 1` and `unfavorable -> 0`; after prediction I mapped the outputs back to COMPAS coding before calculating AIF360 metrics.

## 5. AFTER metrics and trade-off

| Metric | Before | After | Change |
|---|---:|---:|---:|
| Accuracy | 0.664 | 0.652 | -0.012 |
| Disparate impact | 0.773 | 0.950 | +0.177 |
| Statistical parity difference | -0.165 | -0.032 | +0.133 |
| TPR difference | -0.095 | +0.010 | +0.104 |
| FPR difference | -0.183 | -0.012 | +0.172 |
| Equalized-Odds gap | 0.183 | 0.012 | -0.172 |
| Average odds difference | -0.139 | -0.001 | +0.138 |

**Primary-objective reading:** The Equalized-Odds gap decreased on the main split.

**Secondary disparate-impact check:** FAIL (< 0.8): potential disparate impact -> PASS (>= 0.8). This is reported for transparency, but disparate impact was not the optimization target.

## 6. Uncertainty across 10 splits

| Metric | Before mean ± SD | After mean ± SD | Mean change |
|---|---:|---:|---:|
| Accuracy | 0.671 ± 0.013 | 0.664 ± 0.014 | -0.007 |
| Equalized-Odds gap | 0.190 ± 0.034 | 0.059 ± 0.018 | -0.131 |
| TPR difference | -0.091 ± 0.029 | +0.034 ± 0.028 | +0.125 |
| FPR difference | -0.190 ± 0.034 | -0.029 ± 0.047 | +0.161 |
| Disparate impact | 0.775 ± 0.032 | 0.969 ± 0.052 | +0.194 |

The repeated-split analysis determines whether the apparent Equalized-Odds improvement is stable rather than a lucky split. The mean accuracy change is `-0.007`. The mean Equalized-Odds-gap change is `-0.131`. These standard deviations are descriptive stability measures, not formal confidence intervals.

## 7. Recommendation
The deployment recommendation must be based primarily on the achieved TPR/FPR balance, not on disparate impact alone. If Equalized Odds improves consistently across splits at an acceptable accuracy cost, the mitigation is a defensible candidate for further evaluation. It is still not sufficient evidence for high-stakes deployment without jurisdiction-specific validation, calibration analysis, and intersectional subgroup audits.

## 8. Honest limitation
This experiment deliberately changes the fairness objective, but it still audits only race marginally and uses only one mitigation family. ExponentiatedGradient also produces a randomized ensemble, which improves constraint satisfaction at the cost of a less straightforward individual-level explanation than a single deterministic Logistic Regression. I therefore cannot claim that this method is universally preferable to Reweighing; only that it tests a different and defensible fairness objective.

## Transfer to my doctoral research
The methodological lesson is directly relevant to my future XAI study of school-interruption risk in Peru: I should not ask whether a model is simply “fair.” I must first identify the most serious harm in the educational context, choose a fairness metric that corresponds to that harm, report subgroup error rates, and quantify the predictive trade-off across repeated runs.

---
*All numerical values above are generated from this notebook execution. No result is copied from the professor's exemplar.*
