# anomaly_detection.py
from typing import List, Tuple, Optional
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def _find_time_column(df: pd.DataFrame) -> Optional[str]:
    """Return a time-like column name if present, else None."""
    candidates = [c for c in df.columns if c.lower() in ("time", "timestamp", "date", "datetime")]
    return candidates[0] if candidates else None


class AnomalyDetector:
    """
    Multivariate Time Series Anomaly Detection (Isolation Forest backbone).
    - calibrates contamination to keep training scores low (mean <10, max <25) if possible
    - computes percentile-based abnormality_score in the analysis period (0-100)
    - calculates top_feature_1 .. top_feature_7 using z-score + correlation-change signals
    """

    def __init__(self, contamination_candidates: List[float] = None, random_state: int = 42):
        if contamination_candidates is None:
            contamination_candidates = [0.05, 0.02, 0.01, 0.005, 0.001]
        self.cont_candidates = contamination_candidates
        self.random_state = random_state
        self.model = None
        self.scaler = None
        # stored on fit
        self.train_index = None
        self.X_raw = None
        self.X_scaled = None
        self.train_mean_raw = None
        self.train_std_raw = None
        self.train_corr = None

    def preprocess(self, df: pd.DataFrame, feature_list: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Return (X_raw, X_scaled).
        - forward/backward fill missing values
        - drop constant features
        - standardize features (returned as X_scaled)
        """
        X = df[feature_list].copy()

        # handle invalid non-numeric entries: coerce to NaN then ffill/bfill
        X = X.apply(pd.to_numeric, errors="coerce")
        if X.isnull().any().any():
            X = X.fillna(method="ffill").fillna(method="bfill")

        # drop constant features (zero variance)
        nunique = X.nunique(dropna=True)
        constant_features = nunique[nunique <= 1].index.tolist()
        if constant_features:
            X = X.drop(columns=constant_features)

        # If after dropping constants we lose features, handle upstream
        # scaling
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

        # store scaler for potential future use
        self.scaler = scaler
        return X, X_scaled

    def fit_and_calibrate(self, X_train_scaled: pd.DataFrame):
        """
        Fit IsolationForest on X_train_scaled. (Model object set here.)
        This function only fits — calibration occurs externally by trying different contamination values.
        """
        self.model = IsolationForest(
            contamination=0.05,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model.fit(X_train_scaled)

    def _predict_raw_scores(self, model: IsolationForest, X_all_scaled: pd.DataFrame) -> np.ndarray:
        """Return raw anomaly scores (higher = more anomalous)."""
        raw = -model.decision_function(X_all_scaled)
        return raw

    def compute_top_contributors(self, X_raw: pd.DataFrame, n_top: int = 7,
                                 corr_weight: float = 0.5, corr_window: int = 24) -> pd.DataFrame:
        """
        Compute top contributing features per row.
        Combines:
         - abs(z-score) = abs((value - train_mean) / train_std)
         - correlation-change signal per feature (if features <= 50 to avoid heavy compute)
        Returns DataFrame with columns top_feature_1..top_feature_n_top.
        """
        # compute z-scores (handle zero std)
        train_mean = self.train_mean_raw
        train_std = self.train_std_raw.replace(0, np.nan)  # avoid division by zero
        zscores = (X_raw - train_mean) / train_std
        abs_z = zscores.abs().fillna(0)

        # correlation-change contribution
        corr_dev = pd.DataFrame(0.0, index=X_raw.index, columns=X_raw.columns)
        max_features_for_corr = 50
        if len(X_raw.columns) <= max_features_for_corr and corr_window >= 3:
            # compute training correlation matrix (train portion)
            base_corr = self.train_corr  # precomputed
            cols = X_raw.columns.tolist()
            # for each pair compute rolling corr and compare to base_corr
            for i, a in enumerate(cols):
                for b in cols[i + 1:]:
                    # rolling correlation between a and b
                    try:
                        r_roll = X_raw[a].rolling(window=corr_window, min_periods=1).corr(X_raw[b]).fillna(method="bfill").fillna(method="ffill")
                        # absolute deviation from training correlation
                        dev = (r_roll - base_corr.loc[a, b]).abs().fillna(0)
                        # add contribution to both features
                        corr_dev[a] += dev
                        corr_dev[b] += dev
                    except Exception:
                        # in case rolling.corr fails for some pair, skip
                        continue

            # normalize corr_dev per row to 0..1
            row_max = corr_dev.max(axis=1).replace(0, np.nan).fillna(1.0)
            corr_dev_norm = corr_dev.div(row_max, axis=0).fillna(0)
        else:
            corr_dev_norm = corr_dev  # zeros

        # combine contributions: abs_z (z-score) + corr_weight * corr_dev_norm * mean_abs_z (to match scale)
        mean_abs_z = abs_z.mean(axis=1).replace(0, 1.0)
        combined = abs_z.add(corr_weight * corr_dev_norm.mul(mean_abs_z, axis=0), fill_value=0)

        # For each row build top-n list by contribution percentage (>1%) with tie-break alphabetical
        contribs = []
        for idx, row in combined.iterrows():
            row_vals = row.to_dict()
            total = float(sum(row_vals.values()))
            if total <= 0:
                # no contribution: pad empties
                top_feats = [""] * n_top
            else:
                # compute percentage per feature
                perc = {k: (v / total) * 100 for k, v in row_vals.items()}
                # filter >1%
                filtered = [(k, perc[k]) for k in perc if perc[k] > 1.0]
                # sort by -value then alphabetical
                filtered_sorted = sorted(filtered, key=lambda kv: (-kv[1], kv[0]))
                top_feats = [k for k, _ in filtered_sorted][:n_top]
                while len(top_feats) < n_top:
                    top_feats.append("")
            contribs.append(top_feats)

        contrib_df = pd.DataFrame(contribs, columns=[f"top_feature_{i+1}" for i in range(n_top)], index=combined.index)
        return contrib_df

    def prepare_train_stats(self, X_train_raw: pd.DataFrame):
        """Store train mean, std and correlation matrix for later use in attribution."""
        self.train_mean_raw = X_train_raw.mean()
        self.train_std_raw = X_train_raw.std().replace(0, 0.0)
        try:
            self.train_corr = X_train_raw.corr()
        except Exception:
            # fallback: zero matrix
            self.train_corr = pd.DataFrame(0.0, index=X_train_raw.columns, columns=X_train_raw.columns)


def detect_anomalies(input_csv: str, output_csv: str) -> pd.DataFrame:
    """
    Main entry point.

    Args:
        input_csv: path to input CSV (must contain a time column named 'Time' or similar).
        output_csv: path to save output CSV with 8 new columns.

    Returns:
        DataFrame with original columns plus:
          - abnormality_score (0.0 - 100.0)
          - top_feature_1 .. top_feature_7 (strings)
    """
    # ---- load ----
    df = pd.read_csv(input_csv)
    # detect time column
    time_col = _find_time_column(df)
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    else:
        # if no time-like column, create a synthetic index as Time
        df["Time"] = pd.date_range(start="2000-01-01", periods=len(df), freq="H")
        time_col = "Time"

    df = df.sort_values(time_col).reset_index(drop=True)

    # expected training and analysis windows from spec
    train_start = pd.Timestamp("2004-01-01 00:00:00")
    train_end = pd.Timestamp("2004-01-05 23:59:00")
    analysis_start = pd.Timestamp("2004-01-01 00:00:00")
    analysis_end = pd.Timestamp("2004-01-19 07:59:00")

    # check presence of these timestamps in dataset time range
    df_min, df_max = df[time_col].min(), df[time_col].max()
    # If the dataset contains the required range, use it; otherwise fall back to default behavior:
    if not (train_start >= df_min and analysis_end <= df_max):
        # Fallback: use earliest 120 rows as training (approx 120 hours if data hourly) and full dataset for analysis.
        warnings.warn("Dataset does not contain full reference dates. Falling back to using first 120 rows as training.")
        train_mask = df.index < min(len(df), 120)
    else:
        train_mask = (df[time_col] >= train_start) & (df[time_col] <= train_end)

    if train_mask.sum() < 72:
        raise ValueError("Insufficient training data (<72 rows/hours) as required by spec.")

    # select numeric features (exclude time col)
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if time_col in numeric_features:
        numeric_features.remove(time_col)
    if not numeric_features:
        raise ValueError("No numeric features found in dataset.")

    detector = AnomalyDetector()
    # preprocess returns raw and scaled feature matrices
    X_raw, X_scaled = detector.preprocess(df, numeric_features)
    detector.X_raw = X_raw
    detector.X_scaled = X_scaled

    # training subsets (raw & scaled)
    X_train_raw = X_raw.loc[train_mask]
    X_train_scaled = X_scaled.loc[train_mask]
    detector.prepare_train_stats(X_train_raw)

    # calibrate model over contamination candidates to try to meet train mean/max thresholds
    best_result = None
    best_train_mean = float("inf")
    success = False
    for c in detector.cont_candidates:
        model = IsolationForest(contamination=c, random_state=detector.random_state, n_jobs=-1)
        model.fit(X_train_scaled)

        raw_scores = -model.decision_function(X_scaled)  # higher = more anomalous

        # percentile ranking over analysis period (whole dataset) -> 0..100
        pct = pd.Series(raw_scores).rank(method="average", pct=True).values * 100.0

        # slight smoothing to avoid abrupt single-point spikes; center rolling median (window=3)
        pct_series = pd.Series(pct)
        pct_smooth = pct_series.rolling(window=3, center=True, min_periods=1).median().values

        # add tiny noise to avoid exactly 0 scores
        pct_smooth = pct_smooth + np.random.uniform(1e-8, 1e-3, size=len(pct_smooth))

        # evaluate training stats on smoothed scores
        train_scores = pct_smooth[train_mask.values] if isinstance(train_mask, pd.Series) else pct_smooth[np.where(train_mask)[0]]
        train_mean = float(np.mean(train_scores))
        train_max = float(np.max(train_scores))

        # choose the first contamination that meets thresholds
        if train_mean < 10.0 and train_max < 25.0:
            chosen_model = model
            final_scores = pct_smooth
            success = True
            break

        # otherwise keep best (lowest mean)
        if train_mean < best_train_mean:
            best_train_mean = train_mean
            best_result = (model, pct_smooth, train_mean, train_max)

    if not success:
        # fall back to best_result if available
        if best_result is not None:
            chosen_model, final_scores, train_mean, train_max = best_result
            warnings.warn(f"Could not meet training thresholds. Best training mean={train_mean:.2f}, max={train_max:.2f}. Proceeding with best candidate.")
        else:
            # final fallback: fit with default contamination 0.05
            chosen_model = IsolationForest(contamination=0.05, random_state=detector.random_state, n_jobs=-1)
            chosen_model.fit(X_train_scaled)
            raw_scores = -chosen_model.decision_function(X_scaled)
            final_scores = pd.Series(raw_scores).rank(method="average", pct=True).values * 100.0
            final_scores = pd.Series(final_scores).rolling(window=3, center=True, min_periods=1).median().values
            final_scores = final_scores + np.random.uniform(1e-8, 1e-3, size=len(final_scores))
            train_scores = final_scores[train_mask.values] if isinstance(train_mask, pd.Series) else final_scores[np.where(train_mask)[0]]
            train_mean = float(np.mean(train_scores))
            train_max = float(np.max(train_scores))
            warnings.warn("Using default model; training statistics may not meet thresholds.")

    # final_scores is percentiles 0..100 (smoothed)
    abnormality_score = np.round(final_scores.astype(float), 2)

    # compute top contributors using raw feature deviations and correlation-change signal
    # store train stats
    detector.train_index = X_train_raw.index
    detector.prepare_train_stats(X_train_raw)

    topk_df = detector.compute_top_contributors(X_raw, n_top=7, corr_weight=0.5, corr_window=24)

    # assemble output
    df_out = df.copy()
    df_out["abnormality_score"] = abnormality_score
    # ensure exactly 7 columns named top_feature_1..7
    df_out = pd.concat([df_out.reset_index(drop=True), topk_df.reset_index(drop=True)], axis=1)

    # final validation checks & messages
    training_scores = df_out.loc[train_mask, "abnormality_score"] if train_mask.any() else df_out["abnormality_score"].iloc[:0]
    training_mean = float(training_scores.mean()) if len(training_scores) > 0 else float("nan")
    training_max = float(training_scores.max()) if len(training_scores) > 0 else float("nan")

    # warn if training scores not low enough
    if training_mean >= 10.0 or training_max >= 25.0:
        warnings.warn(f"Training-period scores not sufficiently low: mean={training_mean:.2f}, max={training_max:.2f}")

    # Save output
    df_out.to_csv(output_csv, index=False)

    return df_out
