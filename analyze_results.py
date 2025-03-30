import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from visualization import (
    plot_prediction_interval,
    plot_metric_comparison,
    plot_reliability_coverage
)

sns.set_style("white")
plt.style.use("ggplot")


def compute_model_metrics(df, model_name):
    df = df.dropna()
    true = df['TrueDemand']
    pred = df['0.5 Quantile']
    lower = df['0.05 Quantile']
    upper = df['0.95 Quantile']

    mse = np.mean((true - pred)**2)
    rmse = np.sqrt(mse)
    r2 = 1 - np.sum((true - pred)**2) / np.sum((true - np.mean(true))**2)
    nrmse = rmse / np.mean(true)
    mape = np.mean(np.abs((true - pred) / true)) * 100
    reliability = np.mean((true >= lower) & (true <= upper))

    return {
        "Model": model_name,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "NRMSE": nrmse,
        "MAPE": mape,
        "Reliability": reliability
    }, df

def plot_hourly_deviation_and_coverage(results_dict):
    plt.figure(figsize=(10, 3))
    info = {
        "QMLP": "-x",
        "MeanEnsemble": "-s",
        "CQRCTN": "-o",
        "MedianEnsemble": "-s",
    }
    for name, df in results_dict.items():
        df["DemandHour"] = df["Hour of Day"].dt.hour
        df["Errors"] = np.sqrt((df["TrueDemand"] - df["0.5 Quantile"]) ** 2) / df["TrueDemand"].mean()
        avg_errors = df.groupby("DemandHour").mean(numeric_only=True).reset_index()
        if name in info:
            plt.plot(avg_errors["Errors"], info[name], label=name, markersize=5)
    plt.ylabel("Normalized Deviation", fontsize=10)
    plt.xlabel("Hour of Day", fontsize=10)
    plt.legend()
    plt.tight_layout()
    plt.savefig("ErrorByHour.pdf")
    plt.show()

    plt.figure(figsize=(10, 3))
    for name, df in results_dict.items():
        df["DemandHour"] = df["Hour of Day"].dt.hour
        df["Coverage"] = ((df["0.5 Quantile"] > df["0.05 Quantile"]) & (df["0.5 Quantile"] <= df["0.95 Quantile"]))
        avg_coverage = df.groupby("DemandHour").mean(numeric_only=True).reset_index()
        plt.plot(avg_coverage["Coverage"], label=name)
    plt.ylim(0, 1.2)
    plt.xlabel("Hour of Day")
    plt.ylabel("Coverage")
    plt.legend()
    plt.tight_layout()
    plt.savefig("CoverageByHour.pdf")
    plt.show()

def plot_demand_distribution(df):
    plt.figure(figsize=(10, 3))
    df['Total_Demand'].plot()
    plt.title("Time Series of Total Water Demand")
    plt.ylabel("Demand (m³)")
    plt.xlabel("Time Index")
    plt.tight_layout()
    plt.savefig("TotalDemand_Timeseries.pdf")
    plt.show()

    print("Mean:", df["Total_Demand"].mean())
    print("Std Dev:", df["Total_Demand"].std())

def plot_actual_vs_predicted(df_dict):
    from sklearn.metrics import r2_score
    plt.figure(figsize=(10, 5))
    for name, df in df_dict.items():
        plt.scatter(df["TrueDemand"], df["0.5 Quantile"], label=f"{name} (R²={r2_score(df['TrueDemand'], df['0.5 Quantile']):.2f})", s=10, alpha=0.5)
    plt.plot([df["TrueDemand"].min(), df["TrueDemand"].max()], [df["TrueDemand"].min(), df["TrueDemand"].max()], 'k--')
    plt.xlabel("True Demand")
    plt.ylabel("Predicted Demand (0.5 Quantile)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Actual_vs_Predicted.pdf")
    plt.show()

def analyze_results(data_dir="results"):
    model_files = {
        "QReg": "BestQRegPrediction.csv",
        "QMLP": "BestQRNNPrediction.csv",
        "QGradBoost": "BestGradBPrediction.csv",
        "CQRCTN": "CQRCNNBiLSTM_Results.csv",
        "MedianEnsemble": "MedModelPrediction.csv",
        "MeanEnsemble": "MeanModelPrediction.csv"
    }

    metrics = []
    all_preds = {}

    for model_name, file in model_files.items():
        df = pd.read_csv(Path(data_dir) / file)
        result, df_cleaned = compute_model_metrics(df, model_name)
        metrics.append(result)
        all_preds[model_name] = df_cleaned

    metrics_df = pd.DataFrame(metrics)

    plot_metric_comparison(metrics_df, metric_name="NRMSE")
    plot_metric_comparison(metrics_df, metric_name="MAPE")
    plot_reliability_coverage(metrics_df)

    plot_hourly_deviation_and_coverage(all_preds)
    plot_prediction_interval(all_preds['QReg'], 'QReg')
    plot_prediction_interval(all_preds['CQRCTN'], 'CQRCTN')
    plot_actual_vs_predicted(all_preds)

    # Load raw data for distribution plot
    raw_df = pd.read_csv("data/TelfordDemandData.csv", parse_dates=True)
    raw_df["Total_Demand"] = raw_df["Total_Demand"] / 264.1722
    plot_demand_distribution(raw_df)

    return metrics_df

if __name__ == "__main__":
    analyze_results()
