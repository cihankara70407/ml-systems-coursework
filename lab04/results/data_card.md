# Data Card: Census Income Benchmark Dataset

**Author / Maintainer:** Cihan Karayigit (Student ID: 70407)  
**Standard Name:** Adult Census Dataset (OpenML ID: `adult`, version: 2)  
**Application Context:** Predictive Analytics / Supervised Binary Classification  

---

## 1. Dataset Origin & Data Collection
* **Originating Source:** 1994 United States Census Bureau Population Survey database.
* **Extraction Authors:** Ronny Kohavi and Barry Becker.
* **Access Vector:** OpenML REST API / Python Scikit-Learn `fetch_openml(name='adult', version=2)`.
* **Objective:** Predict probability of annual income exceeding $50,000.

---

## 2. Quantitative Dimensions & Partitioning
* **Instance Count:** 48,842 total rows.
* **Feature Count:** 15 variables (14 predictor attributes + 1 binary target `income`).
* **Experimental Splits:**
  * **Training Split (70%):** 34,189 rows pre-deduplication, 34,167 rows post-deduplication (deduplicated solely on train partition to eliminate artificial memorization).
  * **Test Split (30%):** 14,653 rows preserved strictly untouched.
* **Attribute Types:**
  * 6 Continuous Numerical Attributes (`age`, `fnlwgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`).
  * 8 Categorical Attributes (`workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`).
  * 2 Engineered Attributes (`capital_net`, `hours_per_age`).

---

## 3. Licensing & Redistribution
* **License:** Creative Commons Attribution 4.0 International (CC-BY 4.0).
* **Usage Rights:** Open for academic research, education, and comparative benchmarking.

---

## 4. Societal Considerations & Known Biases
1. **Temporal Horizon (1994):** The historical context of the 1994 US workforce exhibits pronounced gender wage disparity (~30.6% of males earn >$50K versus ~10.9% of females in the sample).
2. **Geographical Concentration:** Approximately 90% of observations report `native-country` as the United States, which constrains generalization to modern international demographics.
3. **Sparsity in Financial Features:** Over 91% of individuals record zero capital gains and losses, with a prominent synthetic threshold cap at $99,999.
4. **Class Imbalance:** Significant asymmetry with negative cases (`<=50K`) accounting for ~76% and positive cases (`>50K`) for ~24%.

---

## 5. Anonymization & Privacy Preservation
* **Direct Identifiers:** Names, residential street addresses, and social security numbers were omitted prior to distribution.
* **Demographic Aggregation (`fnlwgt`):** The final sampling weight assigns a population representation factor calculated by the Census Bureau, preventing pinpoint individual re-identification.
* **Ethical Guidance:** Sensitive attributes (`race`, `sex`) must not be used for discriminatory decision-making in lending or hiring without fairness audits (e.g. disparate impact analysis).

---

## 6. Audit Trail & Pipeline Artifacts
* **Fitted Preprocessor:** `artifacts/preprocessor.joblib`
* **Configuration & Checksums:** `configs/pipeline_config.yaml`
* **Execution Command:** `python main.py`
