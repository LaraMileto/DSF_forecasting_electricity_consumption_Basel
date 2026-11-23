import pandas as pd
import numpy as np

# Shared project settings
TRAIN_START = pd.Timestamp("2018-01-01")
MIN_LAG = 5
TARGET = "consumption_mwh"


def validate_data(df):
    """
    Check that the dataset is suitable for chronological
    forecasting and model evaluation.
    """

    # 1. Check required columns
    required_columns = ["date", "period", TARGET]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # 2. Check dates
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        raise ValueError("The date column must be datetime.")

    if df["date"].isna().any():
        raise ValueError("Missing dates found.")

    if not df["date"].is_unique:
        raise ValueError("Duplicate dates found.")

    if not df["date"].is_monotonic_increasing:
        raise ValueError("Dates are not chronologically sorted.")

    # 3. Check period labels
    valid_periods = {"train", "validation", "test"}

    if not df["period"].isin(valid_periods).all():
        raise ValueError("Invalid period labels found.")

    # 4. Check target values
    if df[TARGET].isna().any():
        raise ValueError("Missing target values found.")

    if not np.isfinite(df[TARGET].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite target values found.")

    print("Data validation passed.")
    print(f"Total observations: {len(df)}")

    return True




def get_training_cutoff(first_target_date):
    """
    Return the latest eligible training target date.

    The model is refitted monthly, before the first
    forecast of the new month.

    We assume a conservative five-calendar-day
    availability gap between training targets and
    the first prediction target.
    """

    first_target_date = pd.Timestamp(first_target_date)

    cutoff = first_target_date - pd.Timedelta(days=MIN_LAG)

    return cutoff




def get_expanding_fold(df, prediction_month):
    """
    Select training and validation data for one month
    using an expanding training window.

    Training starts on 2018-01-01 and ends at the
    five-day eligibility cutoff.
    """

    # First day of the prediction month
    month_start = pd.Timestamp(prediction_month).replace(day=1)

    # First day of the following month
    month_end = month_start + pd.DateOffset(months=1)

    # Latest eligible training date
    cutoff = get_training_cutoff(month_start)

    # Select training observations
    train = df[
        (df["date"] >= TRAIN_START) &
        (df["date"] <= cutoff)
    ].copy()

    # Select validation observations
    validation = df[
        (df["date"] >= month_start) &
        (df["date"] < month_end)
    ].copy()

    # Check chronological separation
    if train.empty or validation.empty:
        raise ValueError("Empty training or validation dataset.")

    assert train["date"].max() <= cutoff
    assert train["date"].max() < validation["date"].min()

    return train, validation




def get_rolling_fold(df, prediction_month, years=3):
    """
    Select training and validation data using a
    rolling window of three calendar years.

    The training window ends at the five-day
    eligibility cutoff.
    """

    # First day of the prediction month
    month_start = pd.Timestamp(prediction_month).replace(day=1)

    # First day of the following month
    month_end = month_start + pd.DateOffset(months=1)

    # Latest eligible training date
    cutoff = get_training_cutoff(month_start)

    # First day of the rolling training window
    rolling_start = cutoff - pd.DateOffset(years=years) + pd.Timedelta(days=1)

    # Select training observations
    train = df[
        (df["date"] >= rolling_start) &
        (df["date"] <= cutoff)
    ].copy()

    # Select validation observations
    validation = df[
        (df["date"] >= month_start) &
        (df["date"] < month_end)
    ].copy()

    # Check chronological separation
    if train.empty or validation.empty:
        raise ValueError("Empty training or validation dataset.")

    assert train["date"].min() >= rolling_start
    assert train["date"].max() <= cutoff
    assert train["date"].max() < validation["date"].min()

    return train, validation




def compute_metrics(actual, predicted):
    """
    Calculate common forecasting evaluation metrics.

    Error convention:
    error = actual - predicted

    Returns MAE, RMSE, mean signed error and
    the number of evaluated observations.
    """

    # Convert inputs to numpy arrays
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    # Check dimensions
    if actual.ndim != 1 or predicted.ndim != 1:
        raise ValueError("Inputs must be one-dimensional.")

    # Check equal length
    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted lengths differ.")

    # Check nonempty arrays
    if len(actual) == 0:
        raise ValueError("No observations available.")

    # Check missing or invalid values
    if not np.isfinite(actual).all():
        raise ValueError("Invalid actual values found.")

    if not np.isfinite(predicted).all():
        raise ValueError("Invalid predicted values found.")

    # Calculate prediction errors
    errors = actual - predicted

    # Calculate evaluation metrics
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    mean_error = np.mean(errors)

    return {
        "n_days": len(actual),
        "mae_mwh": float(mae),
        "rmse_mwh": float(rmse),
        "mean_error_mwh": float(mean_error)
    }




def validate_cv_splits(dates, cv):
    """
    Verify that chronological cross-validation splits
    respect the five-calendar-day availability rule.

    Dates must be sorted chronologically.
    """

    dates = pd.DatetimeIndex(dates)

    if dates.hasnans:
        raise ValueError("Missing dates in cross-validation.")

    if not dates.is_monotonic_increasing:
        raise ValueError("Dates must be chronologically sorted.")

    if not dates.is_unique:
        raise ValueError("Duplicate dates in cross-validation.")

    n_splits_checked = 0

    for train_idx, val_idx in cv.split(np.arange(len(dates))):

        last_train_date = dates[train_idx].max()
        first_val_date = dates[val_idx].min()

        required_cutoff = first_val_date - pd.Timedelta(
            days=MIN_LAG
        )

        if last_train_date > required_cutoff:
            raise ValueError(
                f"Data leakage detected: training ends "
                f"{last_train_date.date()}, but must end "
                f"by {required_cutoff.date()}."
            )

        n_splits_checked += 1

    return n_splits_checked




def validate_predictions(predictions, expected_year=2025):
    """
    Validate exported daily prediction results.
    """

    required_columns = [
        "date",
        "model",
        "validation_month",
        "actual_mwh",
        "predicted_mwh",
        "prediction_error_mwh",
        "absolute_error_mwh"
    ]

    # Check required columns
    missing = [
        col for col in required_columns
        if col not in predictions.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df_pred = predictions.copy()
    df_pred["date"] = pd.to_datetime(df_pred["date"])

    # Check missing dates and model names
    if df_pred["date"].isna().any():
        raise ValueError("Missing prediction dates.")

    if df_pred["model"].isna().any():
        raise ValueError("Missing model names.")

    # Each model should have only one prediction per day
    if df_pred.duplicated(["date", "model"]).any():
        raise ValueError("Duplicate model-date combinations.")

    # Check evaluation year
    if not (df_pred["date"].dt.year == expected_year).all():
        raise ValueError("Predictions outside expected year.")

    # Check numerical values
    numeric_columns = [
        "actual_mwh",
        "predicted_mwh",
        "prediction_error_mwh",
        "absolute_error_mwh"
    ]

    for col in numeric_columns:
        if not np.isfinite(
            df_pred[col].to_numpy(dtype=float)
        ).all():
            raise ValueError(f"Invalid values in {col}.")

    # Verify error calculations
    calculated_error = (
        df_pred["actual_mwh"] - df_pred["predicted_mwh"]
    )

    if not np.allclose(
        df_pred["prediction_error_mwh"],
        calculated_error
    ):
        raise ValueError("Incorrect prediction errors.")

    if not np.allclose(
        df_pred["absolute_error_mwh"],
        np.abs(calculated_error)
    ):
        raise ValueError("Incorrect absolute errors.")

    print("Prediction validation passed.")
    print("Models:", df_pred["model"].nunique())
    print("Predictions:", len(df_pred))

    return True




def validate_actual_values(predictions, features):
    """
    Check that actual consumption values in the
    prediction table match the frozen feature dataset.
    """

    reference = features[["date", "consumption_mwh"]].copy()
    reference["date"] = pd.to_datetime(reference["date"])

    pred = predictions[["date", "actual_mwh"]].copy()
    pred["date"] = pd.to_datetime(pred["date"])

    # Match predictions with reference consumption
    merged = pred.merge(
        reference,
        on="date",
        how="left",
        validate="many_to_one"
    )

    # Check whether every prediction date exists
    if merged["consumption_mwh"].isna().any():
        raise ValueError(
            "Some prediction dates are missing in features.csv."
        )

    # Compare actual consumption values
    if not np.allclose(
        merged["actual_mwh"],
        merged["consumption_mwh"],
        rtol=0,
        atol=1e-6
    ):
        raise ValueError(
            "Actual values do not match features.csv."
        )

    print("Actual consumption values match features.csv.")

    return True




def validate_prediction_coverage(predictions, year=2025):
    """
    Check complete daily prediction coverage and
    correct monthly labels for each model.
    """

    pred = predictions.copy()
    pred["date"] = pd.to_datetime(pred["date"])

    expected_dates = pd.date_range(
        start=f"{year}-01-01",
        end=f"{year}-12-31",
        freq="D"
    )

    # Check the monthly labels
    expected_months = pred["date"].dt.strftime("%Y-%m")

    actual_months = (
        pred["validation_month"].astype(str).str[:7]
    )

    if not (expected_months == actual_months).all():
        raise ValueError(
            "Validation month does not match prediction date."
        )

    # Check complete coverage for every model
    for model, group in pred.groupby("model"):
        actual_dates = pd.DatetimeIndex(group["date"])

        missing = expected_dates.difference(actual_dates)
        extra = actual_dates.difference(expected_dates)

        if len(missing) > 0 or len(extra) > 0:
            raise ValueError(
                f"Incomplete coverage for {model}: "
                f"{len(missing)} missing, {len(extra)} extra dates."
            )

        if actual_dates.has_duplicates:
            raise ValueError(
                f"Duplicate prediction dates for {model}."
            )

    print("Prediction coverage validation passed.")
    print("Expected days:", len(expected_dates))

    return True




if __name__ == "__main__":

    from pathlib import Path

    # Find the project root
    ROOT = Path(__file__).resolve().parent.parent

    # Load the existing dataset
    df = pd.read_csv(
        ROOT / "data" / "processed" / "features.csv",
        parse_dates=["date"]
    )

    # Run validation checks
    validate_data(df)




    # Test shared evaluation metrics
    actual = [100, 200, 300]
    predicted = [110, 180, 330]

    metrics = compute_metrics(actual, predicted)

    print("\nEvaluation metrics test")
    print("Observations:", metrics["n_days"])
    print("MAE:", round(metrics["mae_mwh"], 2))
    print("RMSE:", round(metrics["rmse_mwh"], 2))
    print("Mean Error:", round(metrics["mean_error_mwh"], 2))

    # Verify expected results
    assert metrics["n_days"] == 3
    assert np.isclose(metrics["mae_mwh"], 20.0)
    assert np.isclose(metrics["rmse_mwh"], 21.602468)
    assert np.isclose(metrics["mean_error_mwh"], -6.666667)




    # Test rolling three-year training window
    train_roll, val_roll = get_rolling_fold(df, "2025-01-01")

    print("\nRolling window – January 2025")
    print("Training start:", train_roll["date"].min().date())
    print("Training end:", train_roll["date"].max().date())
    print("Validation start:", val_roll["date"].min().date())
    print("Validation end:", val_roll["date"].max().date())
    print("Training observations:", len(train_roll))
    print("Validation observations:", len(val_roll))

    # Verify expected boundaries
    assert train_roll["date"].min() == pd.Timestamp("2021-12-28")
    assert train_roll["date"].max() == pd.Timestamp("2024-12-27")
    assert len(val_roll) == 31




    # Test expanding training window
    train, validation = get_expanding_fold(df, "2025-02-01")

    print("\nExpanding window – February 2025")
    print("Training start:", train["date"].min().date())
    print("Training end:", train["date"].max().date())
    print("Validation start:", validation["date"].min().date())
    print("Validation end:", validation["date"].max().date())
    print("Training observations:", len(train))
    print("Validation observations:", len(validation))




    # Test monthly training cutoff
    cutoff = get_training_cutoff("2025-02-01")

    print("February 2025 training cutoff:", cutoff.date())

    assert cutoff == pd.Timestamp("2025-01-27")




    from sklearn.model_selection import TimeSeriesSplit

    # Use only historical training data
    historical_dates = df.loc[
        df["date"] <= pd.Timestamp("2024-12-27"),
        "date"
    ]

    cv = TimeSeriesSplit(n_splits=5, gap=4)

    checked = validate_cv_splits(historical_dates, cv)

    print("\nCross-validation test")
    print("Splits checked:", checked)
    print("All splits respect the five-day rule.")




    # Test prediction validation with baseline results
    baseline_path = (
        ROOT / "output" / "tables"
        / "baseline_validation_predictions.csv"
    )

    baseline_predictions = pd.read_csv(baseline_path)

    print("\nBaseline prediction validation")
    validate_predictions(baseline_predictions)


    # Verify actual values against the frozen dataset
    validate_actual_values(baseline_predictions, df)




    # Reproduce baseline evaluation metrics
    baseline_metrics = compute_metrics(
        baseline_predictions["actual_mwh"],
        baseline_predictions["predicted_mwh"]
    )

    print("\nBaseline metrics reproduction")
    print("Observations:", baseline_metrics["n_days"])
    print("MAE:", round(baseline_metrics["mae_mwh"], 2))
    print("RMSE:", round(baseline_metrics["rmse_mwh"], 2))
    print(
        "Mean Error:",
        round(baseline_metrics["mean_error_mwh"], 2)
    )

    # Compare with previously reported baseline results
    assert baseline_metrics["n_days"] == 365
    assert np.isclose(
        baseline_metrics["mae_mwh"], 137.3, atol=0.1
    )
    assert np.isclose(
        baseline_metrics["rmse_mwh"], 201.5, atol=0.1
    )

    print("Baseline metrics reproduction passed.")




    # Test complete 2025 prediction coverage
    print("\nBaseline coverage test")

    validate_prediction_coverage(
        baseline_predictions,
        year=2025
    )




    # Test linear model predictions
    linear_path = (
        ROOT / "output" / "tables"
        / "linear_models_validation_predictions.csv"
    )

    linear_predictions = pd.read_csv(linear_path)

    print("\nLinear models validation")

    validate_predictions(linear_predictions)

    validate_actual_values(linear_predictions, df)

    validate_prediction_coverage(
        linear_predictions,
        year=2025
    )

    print(
        "Linear model names:",
        sorted(linear_predictions["model"].unique())
    )




    # Validate Random Forest predictions
    rf_path = (
        ROOT / "output" / "tables"
        / "random_forest_validation.csv"
    )

    rf_predictions = pd.read_csv(rf_path)

    print("\nRandom Forest validation")

    validate_predictions(rf_predictions)

    validate_actual_values(rf_predictions, df)

    validate_prediction_coverage(
        rf_predictions,
        year=2025
    )

    print(
        "Model names:",
        sorted(rf_predictions["model"].unique())
    )




    # Validate tuned Random Forest predictions
    rf_tuned_path = (
        ROOT / "output" / "tables"
        / "random_forest_tuned_validation.csv"
    )

    rf_tuned_predictions = pd.read_csv(rf_tuned_path)

    print("\nTuned Random Forest validation")

    validate_predictions(rf_tuned_predictions)

    validate_actual_values(rf_tuned_predictions, df)

    validate_prediction_coverage(
        rf_tuned_predictions,
        year=2025
    )

    print(
        "Model names:",
        sorted(rf_tuned_predictions["model"].unique())
    )




    # Validate training window predictions
    windows_path = (
        ROOT / "output" / "tables"
        / "training_windows_validation.csv"
    )

    windows_predictions = pd.read_csv(windows_path)

    print("\nTraining windows validation")

    validate_predictions(windows_predictions)

    validate_actual_values(windows_predictions, df)

    validate_prediction_coverage(
        windows_predictions,
        year=2025
    )

    print(
        "Model names:",
        sorted(windows_predictions["model"].unique())
    )




    # Recalculate metrics for all validated models
    print("\nFinal validation metrics")

    all_predictions = pd.concat(
        [
            baseline_predictions,
            linear_predictions,
            rf_predictions,
            rf_tuned_predictions,
            windows_predictions
        ],
        ignore_index=True
    )

    for (source, model), group in all_predictions.assign(
        source=(
            ["baseline"] * len(baseline_predictions)
            + ["linear"] * len(linear_predictions)
            + ["random_forest"] * len(rf_predictions)
            + ["random_forest_tuned"] * len(rf_tuned_predictions)
            + ["training_windows"] * len(windows_predictions)
        )
    ).groupby(["source", "model"]):

        metrics = compute_metrics(
            group["actual_mwh"],
            group["predicted_mwh"]
        )

        print(
            f"{source} | {model}: "
            f"MAE={metrics['mae_mwh']:.2f}, "
            f"RMSE={metrics['rmse_mwh']:.2f}"
        )
