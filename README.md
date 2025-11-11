# Gradient Boosting Ad-Click Model

## Process
- Clean `data/ad_click.csv`: normalize categorical strings, clip ages to [18, 80], add `age_missing` and coarse `age_bucket`.
- Split data into stratified 80/20 train-test sets (`random_state=42`).
- Apply median imputation to numeric fields and constant-fill + dense one-hot encodings to categoricals via a `ColumnTransformer`.
- Train `HistGradientBoostingClassifier(learning_rate=0.08, max_iter=400, max_depth=6, min_samples_leaf=20, l2_regularization=1.0, random_state=42)`.
- Evaluate the held-out fold to obtain accuracy **0.7285**.

## Reproduce Results
1. `cd interactive-510`
2. (Optional) activate the Python environment containing scikit-learn ≥ 1.0.
3. Run `python train_gradient_boosting.py`.
4. The script loads the CSV, trains the pipeline, and prints the accuracy.
