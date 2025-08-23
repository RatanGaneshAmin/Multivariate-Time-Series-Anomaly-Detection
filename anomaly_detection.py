# anomaly_detection.py
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", rc={"figure.figsize": (12, 5)})

def detect_anomalies(df_full, feature_list, output_csv,
                     contamination=0.05, train_frac=0.7, random_state=42):
    """
    df_full: original dataframe (may include timestamp and other cols)
    feature_list: list of numeric feature column names to use
    output_csv: path to save results (CSV)
    Returns: df_result (original columns + anomaly_score, anomaly, top_feature)
    """
    if len(feature_list) == 0:
        raise ValueError("feature_list is empty. Provide numeric columns to analyze.")

    # Copy features only (safe)
    X = df_full[feature_list].copy()

    # Simple cleaning
    if X.isnull().any().any():
        X = X.fillna(X.median())  # simple imputation

    # Scale features (important for many datasets)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    n = len(X_scaled)
    train_size = max(1, int(train_frac * n))

    X_train = X_scaled.iloc[:train_size]
    X_all = X_scaled

    # Fit model
    model = IsolationForest(contamination=contamination, random_state=random_state)
    model.fit(X_train)

    # get anomaly score (higher = more anomalous)
    raw_scores = -model.decision_function(X_all)
    # scale to 0-100
    if raw_scores.max() - raw_scores.min() == 0:
        scaled_scores = np.zeros_like(raw_scores)
    else:
        scaled_scores = 100.0 * (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())

    # predictions (1 normal, -1 anomaly) -> convert to 0/1 or labels
    preds = pd.Series(model.predict(X_all), index=X_all.index)
    label_map = {1: "Normal", -1: "Anomaly"}
    labels = preds.map(label_map)

    # Feature contributions: absolute deviation from training mean (on original scale)
    means_train = X.iloc[:train_size].mean()
    contributions = np.abs(X - means_train)  # original scale deviations
    top_features = contributions.idxmax(axis=1)

    # assemble results: keep original df and add new cols
    df_result = df_full.copy()
    df_result["anomaly_score"] = np.round(scaled_scores, 2)
    df_result["anomaly"] = labels.values
    df_result["top_feature"] = top_features.values

    # save results
    df_result.to_csv(output_csv, index=False)

    return df_result


def plot_anomalies(df_result, feature_list, timestamp_col=None,
                   anomaly_col="anomaly", score_col="anomaly_score",
                   out_plot="anomaly_plot.png", out_feat="feature_importance.png"):
    """
    Plots first up to 3 features in feature_list; uses timestamp_col if provided.
    Produces:
      - out_plot: time series plot with anomalies highlighted
      - out_feat: bar/count plot of top contributing features
    """
    # Decide x-axis
    if timestamp_col and timestamp_col in df_result.columns:
        x = pd.to_datetime(df_result[timestamp_col], errors='coerce')
        x_is_valid = not x.isnull().all()
        if x_is_valid:
            x_vals = x
        else:
            x_vals = df_result.index
    else:
        x_vals = df_result.index

    # choose up to first 3 features to plot
    plot_features = feature_list[:3]

    plt.close("all")
    fig, axes = plt.subplots(len(plot_features), 1, figsize=(14, 3.5 * len(plot_features)), sharex=True)
    if len(plot_features) == 1:
        axes = [axes]

    anomalies_mask = df_result[anomaly_col] == "Anomaly"

    for ax, feat in zip(axes, plot_features):
        ax.plot(x_vals, df_result[feat], label=feat, linewidth=1.2)
        # plot anomaly points
        ax.scatter(x_vals[anomalies_mask], df_result.loc[anomalies_mask, feat],
                   color="red", s=40, marker="x", label="Anomaly")
        ax.set_ylabel(feat)
        ax.legend(loc="upper left")
        ax.grid(True)

    # add a small subplot for anomaly_score if more than 1 feature
    if len(plot_features) >= 1:
        axes[-1].set_xlabel("Time" if isinstance(x_vals, (pd.DatetimeIndex, pd.Series)) else "Index")

    fig.suptitle("Time Series (top features) with Detected Anomalies", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_plot)
    plt.close(fig)

    # Feature importance plot (counts)
    plt.figure(figsize=(8, 5))
    sns.countplot(y="top_feature", hue=anomaly_col, data=df_result,
                  order=df_result["top_feature"].value_counts().index)
    plt.title("Top Contributing Features (count by Anomaly/Normal)")
    plt.tight_layout()
    plt.savefig(out_feat)
    plt.close()

