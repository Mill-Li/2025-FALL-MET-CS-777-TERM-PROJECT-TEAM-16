import sys
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline

from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier,
    GBTClassifier,
    OneVsRest,
)

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics

def gcs_to_local_path(gcs_path: str) -> str:
    if gcs_path.startswith("gs://"):
        no_gs = gcs_path[5:]
        parts = no_gs.split("/", 1)
        bucket = parts[0]
        rest = "/" + parts[1] if len(parts) > 1 else ""
        return f"/gcs/{bucket}{rest}"
    else:
        return gcs_path


# --------------------------------------------------------------------
# 1) Data preprocessing: Read in parquet -> Filter mode -> Clean features -> Generate final_df
# --------------------------------------------------------------------
def prepare_data(spark: SparkSession, input_path: str):
    print(f"[INFO] Reading parquet from: {input_path}")
    df = spark.read.parquet(input_path)
    modes_to_keep = ["walk", "bus", "bike", "car", "subway", "train"]
    df1 = df.filter(col("mode").isin(modes_to_keep))
    feature_cols = [
        "total_distance_m",
        "max_speed",
        "median_speed",
        "var_speed",
        "mean_accel",
        "max_accel",
        "stop_duration_seconds",
        "duration_seconds",
        "mean_speed_calculated",
    ]
    df2_clean = df1.dropna(subset=feature_cols)
    df2_clean = df2_clean.filter(col("duration_seconds") > 10)
    df2_clean = df2_clean.filter(col("total_distance_m") > 10)

    print("[INFO] After cleaning, row count =", df2_clean.count())
    label_indexer = StringIndexer(
        inputCol="mode", outputCol="label", handleInvalid="skip"
    )
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
    scaler = StandardScaler(
        inputCol="features_raw", outputCol="features", withMean=True, withStd=True
    )

    prep_pipeline = Pipeline(stages=[label_indexer, assembler, scaler])
    prep_model = prep_pipeline.fit(df2_clean)

    final_df = prep_model.transform(df2_clean).select("features", "label", "mode")

    print("[INFO] final_df schema:")
    final_df.printSchema()

    return final_df, assembler

# --------------------------------------------------------------------
# 2) Train 3 models and return the prediction results and metrics.
# --------------------------------------------------------------------
def train_and_evaluate_models(final_df):
    train_df, test_df = final_df.randomSplit([0.8, 0.2], seed=42)
    print("[INFO] Train count:", train_df.count())
    print("[INFO] Test  count:", test_df.count())
    acc_eval = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )
    f1_eval = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    )
    metrics = {}
    preds = {}
    # ---- Logistic Regression ----
    print("[INFO] Training Logistic Regression...")
    lr = LogisticRegression(
        labelCol="label",
        featuresCol="features",
        maxIter=100,
        regParam=0.01,
        elasticNetParam=0.0,  # L2
    )
    lr_model = lr.fit(train_df)
    lr_pred = lr_model.transform(test_df)
    lr_acc = acc_eval.evaluate(lr_pred)
    lr_f1 = f1_eval.evaluate(lr_pred)
    print(f"[RESULT] Logistic Regression  Accuracy = {lr_acc:.4f}")
    print(f"[RESULT] Logistic Regression  F1       = {lr_f1:.4f}")
    metrics["Logistic Regression"] = (lr_acc, lr_f1)
    preds["Logistic Regression"] = lr_pred
    # ---- Random Forest ----
    print("[INFO] Training Random Forest...")
    rf = RandomForestClassifier(
        labelCol="label",
        featuresCol="features",
        numTrees=200,
        maxDepth=12,
        seed=42,
    )
    rf_model = rf.fit(train_df)
    rf_pred = rf_model.transform(test_df)
    rf_acc = acc_eval.evaluate(rf_pred)
    rf_f1 = f1_eval.evaluate(rf_pred)
    print(f"[RESULT] Random Forest         Accuracy = {rf_acc:.4f}")
    print(f"[RESULT] Random Forest         F1       = {rf_f1:.4f}")
    metrics["Random Forest"] = (rf_acc, rf_f1)
    preds["Random Forest"] = rf_pred
    # ---- Gradient-Boosted Trees ----
    print("[INFO] Training GBT (OneVsRest)...")
    base_gbt = GBTClassifier(
        labelCol="label",
        featuresCol="features",
        maxIter=80,
        maxDepth=6,
        stepSize=0.1,
        seed=42,
    )
    ovr = OneVsRest(
        classifier=base_gbt,
        labelCol="label",
        featuresCol="features",
    )
    gbt_ovr_model = ovr.fit(train_df)
    gbt_pred = gbt_ovr_model.transform(test_df)
    gbt_acc = acc_eval.evaluate(gbt_pred)
    gbt_f1 = f1_eval.evaluate(gbt_pred)
    print(f"[RESULT] GBT (OneVsRest)       Accuracy = {gbt_acc:.4f}")
    print(f"[RESULT] GBT (OneVsRest)       F1       = {gbt_f1:.4f}")
    metrics["Gradient-Boosted Trees"] = (gbt_acc, gbt_f1)
    preds["Gradient-Boosted Trees"] = gbt_pred
    return metrics, preds, rf_model

# --------------------------------------------------------------------
# 3) Calculate & Save Confusion Matrix (PNG)
# --------------------------------------------------------------------
def save_confusion_matrix(pred_df, model_name, out_png_path):
    import matplotlib.pyplot as plt
    import numpy as np
    labels = ["walk", "bus", "bike", "car", "subway", "train"]

    rdd = pred_df.select("prediction", "label").rdd.map(tuple)
    metrics = MulticlassMetrics(rdd)
    cm = metrics.confusionMatrix().toArray()

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                int(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_png_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved confusion matrix for {model_name} to {out_png_path}")

# --------------------------------------------------------------------
# 4) Save the feature importance graph of Random Forest
# --------------------------------------------------------------------
def save_feature_importance(rf_model, feature_names, out_png_path):
    import matplotlib.pyplot as plt
    import pandas as pd

    importance = rf_model.featureImportances.toArray()
    df = pd.DataFrame({"feature": feature_names, "importance": importance})
    df = df.sort_values("importance", ascending=False)

    plt.figure(figsize=(8, 5))
    plt.barh(df["feature"], df["importance"])
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_png_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved feature importance to {out_png_path}")

# --------------------------------------------------------------------
# 5) Comparison Bar Chart of Saved Models
# --------------------------------------------------------------------
def save_model_comparison_barplots(metrics_dict, out_prefix_png):
    import matplotlib.pyplot as plt
    import pandas as pd

    rows = []
    for name, (acc, f1) in metrics_dict.items():
        rows.append({"Model": name, "Accuracy": float(acc), "F1": float(f1)})

    df = pd.DataFrame(rows)
    # Accuracy
    plt.figure(figsize=(7, 5))
    plt.bar(df["Model"], df["Accuracy"])
    plt.xticks(rotation=30, ha="right")
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    acc_png = out_prefix_png + "_accuracy.png"
    plt.savefig(acc_png, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved accuracy comparison to {acc_png}")

    # F1
    plt.figure(figsize=(7, 5))
    plt.bar(df["Model"], df["F1"])
    plt.xticks(rotation=30, ha="right")
    plt.title("Model F1 Score Comparison")
    plt.ylabel("F1")
    plt.tight_layout()
    f1_png = out_prefix_png + "_f1.png"
    plt.savefig(f1_png, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved F1 comparison to {f1_png}")
    return df
# --------------------------------------------------------------------
# 6) Main function
# --------------------------------------------------------------------
def main(input_path: str, output_prefix_gcs: str):
    # SparkSession
    spark = (
        SparkSession.builder.appName("Geolife-Mode-Classification-Full")
        .getOrCreate()
    )
    final_df, assembler = prepare_data(spark, input_path)
    metrics, preds, rf_model = train_and_evaluate_models(final_df)
    rows = []
    for name, (acc, f1) in metrics.items():
        rows.append({"model": name, "accuracy": float(acc), "f1": float(f1)})

    metrics_df = spark.createDataFrame(rows)
    metrics_out_gcs = output_prefix_gcs + "/metrics_models"
    (
        metrics_df.coalesce(1)
        .write.mode("overwrite")
        .option("header", True)
        .csv(metrics_out_gcs)
    )
    print(f"[INFO] Saved metrics CSV to {metrics_out_gcs}")
    plots_root_local = gcs_to_local_path(output_prefix_gcs) + "/plots"
    os.makedirs(plots_root_local, exist_ok=True)
    print(f"[INFO] Plots will be saved under local path: {plots_root_local}")
    save_confusion_matrix(
        preds["Random Forest"],
        "Random Forest",
        os.path.join(plots_root_local, "confusion_rf.png"),
    )
    save_confusion_matrix(
        preds["Gradient-Boosted Trees"],
        "GBT (OneVsRest)",
        os.path.join(plots_root_local, "confusion_gbt.png"),
    )
    save_confusion_matrix(
        preds["Logistic Regression"],
        "Logistic Regression",
        os.path.join(plots_root_local, "confusion_lr.png"),
    )
    save_feature_importance(
        rf_model,
        assembler.getInputCols(),
        os.path.join(plots_root_local, "feature_importance_rf.png"),
    )
    comparison_df = save_model_comparison_barplots(
        metrics,
        os.path.join(plots_root_local, "model_comparison"),
    )
    comparison_spark_df = spark.createDataFrame(comparison_df)
    comparison_out_gcs = output_prefix_gcs + "/metrics_comparison"
    (
        comparison_spark_df.coalesce(1)
        .write.mode("overwrite")
        .option("header", True)
        .csv(comparison_out_gcs)
    )
    print(f"[INFO] Saved comparison CSV to {comparison_out_gcs}")

    spark.stop()
    print("[INFO] Job finished successfully.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: geolife_mode_classification.py "
            "<input_parquet_gs_path> <output_prefix_gs_path>",
            file=sys.stderr,
        )
        sys.exit(1)

    input_parquet = sys.argv[1]
    output_prefix = sys.argv[2]
    main(input_parquet, output_prefix)
