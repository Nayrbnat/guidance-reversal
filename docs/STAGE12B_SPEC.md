# Stage 12b: does the Stage 12 null survive other model families? XGBoost, a deep neural network and support vector machines (registered 2026-11-08, before any Stage 12b result was computed)

**Status:** REGISTERED. What I looked at before writing it: the full Stage 12 results (`STAGE12_SPEC.md`, Section 9), which used gradient boosting and logistic regression. No Stage 12b model has been fitted. This stage is **exploratory**: it reuses the Stage 12 events, features, windows and outcome, and every year was seen in earlier stages.

## 1. What this tests

Stage 12 found that adding the Guidance Reversal ingredients does not improve out-of-sample prediction of large losses (gradient boosting, depth 3: AUC difference -0.018, permutation p 0.945). A reasonable objection is that depth-3 boosting is too simple, or the wrong kind of model, to find a pattern that exists. This stage answers it by repeating the identical test with four other model families. It asks one question: **does any model family find that the ingredients improve out-of-sample prediction?**

## 2. What is held fixed (identical to Stage 12, P12a)

Sample (3,010 modelling events, leak guard, 2004 to 2024), outcome (12-month market-adjusted return below -30 points), features (B baseline, R reversal ingredients; the same columns), rolling windows (train years t-6 to t-2, embargo t-1, test years 2010 to 2024), statistic (mean within-year AUC of B+R minus B, years with at least 5 large losses), and inference (permutation test that shuffles the R rows within announcement year and re-runs the whole rolling procedure; 95% interval from a bootstrap over announcement quarters with predictions fixed). Inputs are standardised with training-window means and standard deviations; missing values are replaced by the training-window median.

## 3. Models (fixed now, no tuning, seed 0)

| Key | Model | Settings |
|---|---|---|
| `xgb` | XGBoost, `XGBClassifier` | 100 trees, depth 3, learning rate 0.05, subsample 0.8, column sample 0.8, minimum child weight 5, L2 penalty 1.0, histogram trees, one thread |
| `mlp` | Deep neural network, scikit-learn `MLPClassifier` | three hidden layers of 64, 32 and 16 units, ReLU, Adam, learning rate 0.001, L2 penalty 0.1, 200 iterations, no early stopping |
| `svm_lin` | Linear support vector machine, `LinearSVC` | C = 0.1, squared hinge loss, L2 penalty; the AUC uses the signed distance to the margin |
| `svm_poly` | Polynomial-kernel support vector machine, `SVC` | degree 2, C = 0.1, coef0 = 1, gamma "scale"; the AUC uses the signed distance to the margin |

**No radial-basis-function kernel**: it is slower and, with about 700 training events, adds nothing a degree-2 polynomial does not. "Deep" here means three hidden layers; this is a small network by modern standards, as the data allow (about 700 training events and about 40 inputs). A larger network would only overfit more.

## 4. Tests

**P12b-i to P12b-iv.** For each model, the Stage 12 statistic: out-of-sample mean within-year AUC of B+R minus B alone, H: positive, one-sided, with a 200-draw permutation p-value (p = (1 + number at least as large) / 201). Reported for every model: AUC of B, AUC of B+R, the difference, the 95% bootstrap interval, the permutation null mean and 95th percentile, and the difference by test year.

**Multiple testing.** The four p-values form one exploratory family; Holm adjustment is applied across the four. The Stage 12 confirmatory test (P12a, boosting) stays the only confirmatory test of this analysis and is not changed. These four are **outside the formal test count** (as the Stage 12 secondaries were), and every one is reported whatever the result.

## 5. Interpretation rules (fixed now)

- **Supported for a model:** a positive AUC difference of at least 0.02, a Holm-adjusted p below 0.05, **and** a positive difference for the Stage 12 logistic benchmark. Anything else is not supported, reported with its interval.
- **If every model is not supported**, the conclusion is that the Stage 12 null is not specific to the model family.
- **If one model is supported**, it is a single exploratory result among four, to be checked on new data and never presented as a finding.
- Reporting the level of AUC (not only the difference) shows whether a more flexible model predicts large losses better from the baseline alone.
- Known limits fixed now: about 700 training events and about 25 large losses per test year among flagged firms; the neural network and the polynomial SVM can overfit and are not tuned; the first-identified announcement date is used for the features, as in Stage 12; the return is market-adjusted.

## 6. Addendum: a compact feature set (registered 2026-11-08, before it was run)

**Why.** After the first three Stage 12b models finished, every model showed a lower out-of-sample AUC when the 19 reversal features were added. A diagnostic (not a registered test) found: in-sample AUC rises by 0.03 while out-of-sample AUC falls by 0.016 (overfitting); heavier ridge shrinkage reduces the loss but never makes it positive; the 7 exit-type dummies are exhaustive (collinear with the intercept) and two sale-value features overlap (VIF above 100); each reversal feature alone has a within-year AUC between 0.45 and 0.55. The objection to answer: the drop may come from clutter and redundancy, not from the ingredients.

**What I had already seen when choosing the set:** the full Stage 12 and 12b results, the one-at-a-time out-of-sample AUC changes and the single-feature AUCs above. To keep the set from being chosen on those results, **it is chosen from the hypothesis alone**: the ingredients the Guidance Reversal definition names and nothing else.

**Compact set R6 (fixed now):** `n_guid_12m` (guidance issued), `sold_post_guid_pct` (sale after guidance, share of market cap), `opt_last` (optimism of the latest guidance), `n_unfulfilled` (guidance whose period ends after the exit), `abrupt_rule` (abrupt exit), `flag_f` (the flag). No exit-type dummies, no second sale-value measure.

**Tests (exploratory, outside the formal count).** The Stage 12 statistic with B + R6 against B alone, same windows, outcome and permutation design, for **all six models**: logistic regression (C = 1, 1,000 permutations), gradient boosting (the Stage 12 settings), XGBoost, the linear SVM and the polynomial SVM (200 permutations each), and the neural network (**100 permutations**, reduced from 200 because each permutation refits the network 15 times; fixed now, before any compact-set result). Supported only under the Section 5 rule (difference of at least 0.02, p below 0.05, and positive for both the logistic and one other model); otherwise not supported. A positive-but-small result would not be presented as a finding.

**The variance-tradeoff comparison (the question this addendum answers).** For each model, the AUC difference (B+R minus B) with the 19 features against the same difference with the 6 features. The variance explanation predicts that the loss shrinks toward zero with fewer features (fewer parameters fitted to noise) and, if the ingredients carried real information, that the difference would turn positive. A loss that shrinks but stays at or below zero means the features add variance and no signal; a loss that does not shrink would point to a different cause (a non-stationary relationship, or a problem I have not found). Reported for all models with the 95% intervals, whatever the pattern.

**Prediction recorded before running:** the difference will be close to zero, not clearly positive; a smaller loss than the 19-feature set is expected because there are fewer parameters to overfit.

## 7. Results

Run 2026-11-08 to 2026-11-09 (`python scripts/run_stage12b.py --kind <model> [--compact]`; `outputs/results/stage12b_*.json`). Same 3,010 events and 2,096 out-of-sample predictions as Stage 12; 15 test years.

**Verdict: not supported for any model, on either feature set. The ingredients do not improve out-of-sample prediction of large losses.**

Change in out-of-sample mean within-year AUC from adding the reversal features to the baseline (95% bootstrap interval; permutation p, one-sided in the hypothesised positive direction):

| Model | 19 features | 6 features (R6) | Permutation p (R6) |
|---|---|---|---|
| Logistic (Stage 12) | -0.017 | -0.006 (-0.015 to +0.010) | 0.32 (1,000 reps) |
| Gradient boosting (Stage 12) | -0.018 | -0.012 (-0.026 to +0.005) | 0.91 |
| XGBoost | -0.028 (-0.043 to -0.001) | -0.009 (-0.022 to +0.006) | 0.77 |
| Linear SVM | -0.016 (-0.031 to +0.005) | -0.004 (-0.013 to +0.011) | 0.27 |
| Polynomial SVM | -0.015 (-0.049 to +0.027) | +0.002 (-0.028 to +0.041) | 0.46 |
| Neural network (3 hidden layers) | -0.015 (-0.040 to +0.023) | -0.012 (-0.031 to +0.019) | 0.050 (100 reps) |

The baseline AUC is 0.62 to 0.63 for logistic, boosting, XGBoost and the linear SVM, 0.58 for the neural network and 0.55 for the polynomial SVM.

**The variance-tradeoff question.** With fewer features the loss shrinks in all six models: markedly for logistic (-0.017 to -0.006), XGBoost (-0.028 to -0.009) and the linear SVM (-0.016 to -0.004), modestly for gradient boosting and the neural network, and to +0.002 for the polynomial SVM. No model turns clearly positive, and every interval includes zero. This matches the prediction recorded in Section 6: the extra features add variance and no out-of-sample signal. It is consistent with, not a formal test of, a variance explanation; the earlier logistic diagnostic (in-sample AUC up 0.03 while out-of-sample AUC fell 0.016) points the same way.

**The neural-network permutation p-values (0.035 with 19 features; 0.050 with 6) are not a finding.** The permutation null for the network is very negative (mean -0.043 with 19 features), because shuffled features hurt a flexible network much more than they hurt the other models; the real features cost -0.015, less than noise does, hence the low p-value. The difference is still negative, below the 0.02 threshold, its bootstrap interval includes zero, and the Holm-adjusted p across the four new models is 0.14 (19 features) and 0.20 (6 features). Under the Section 5 rule it is not supported. The network is also the weakest baseline model, so it is the least informative.

**Limits.** The six-feature set was chosen from the hypothesis after I had seen the first results (disclosed in Section 6); no model was tuned; the polynomial SVM and the network have weak baselines; the neural-network compact run used 100 permutations. All years were seen in earlier stages, so a positive result would have been exploratory.

## Amendment 1 (2026-11-09): returns through 2025-12-31

**Data only.** The legacy CRSP daily file and the market index in the main cache stop on 2024-12-31 on this account (WRDS has no 2026 CRSP rows). The CRSP v2 daily file (`gg_cache/ciz_dsf_2023_2025.pkl`, through 2025-12-31) is appended for later dates (`src/gg_extend.py`); the market return after 2024 is the Fama-French market (mktrf + rf), which over 2022 to 2024 correlates 0.998 with the cached CRSP index and runs about 1.4% a year higher. Nothing else changes: the models, both feature sets, windows, test years and tests. Every Stage 12b run is repeated (`python scripts/run_stage12b.py --extend --both --kind <model>`; `stage12b_ext_*.json`), with 200 permutations (1,000 for logistic regression, 100 for the neural network, as before). The original (2004 to 2024 returns) results stay in this document. I knew those results when I made this change; the change is made because the 2025 returns exist, not because of any result.
