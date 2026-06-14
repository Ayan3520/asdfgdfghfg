"""
House Price Prediction — Machine Learning Pipeline
Educational Practice 2025-2026, Astana IT University
Academic Group BDA-2501, Machine Learning Track

Team:
  Olen       — Team Leader, project management, GitHub workflow, core ML regression
  Kenesary   — Data ingestion, cleaning, outliers, missing values, feature engineering, EDA
  Ayan       — Evaluation metrics R²/RMSE/MAE, final report, feature importance interpretation

Dataset features: Area_sqft, Bedrooms, Bathrooms, Year_Built, Location_Grade, Price
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")
np.random.seed(42)

# ---------------------------------------------------------------------------
# 1. DATA GENERATION / LOADING
# ---------------------------------------------------------------------------

def generate_dataset(n_samples: int = 2000) -> pd.DataFrame:
    """Generate a realistic synthetic house-price dataset.

    The data follows plausible real-estate relationships:
      - Price correlates positively with area, bedrooms, bathrooms, location grade.
      - Newer houses (higher Year_Built) tend to cost more.
      - Non-linear interactions and natural noise are included.
    """
    area = np.random.normal(1500, 500, n_samples).clip(300, 5000)
    bedrooms = np.random.choice([1, 2, 3, 4, 5, 6], n_samples,
                                p=[0.05, 0.15, 0.30, 0.30, 0.15, 0.05])
    bathrooms = np.clip(
        bedrooms + np.random.choice([-1, 0, 0, 1], n_samples, p=[0.1, 0.3, 0.4, 0.2]),
        1, None
    )
    year_built = np.random.randint(1950, 2024, n_samples)
    location_grade = np.random.choice([1, 2, 3, 4, 5], n_samples,
                                      p=[0.05, 0.15, 0.30, 0.30, 0.20])

    # Base price driven by area and location
    price = (area * 120
             + bedrooms * 8000
             + bathrooms * 12000
             + (year_built - 1950) * 350
             + location_grade * 45000
             + np.random.normal(0, 15000, n_samples))  # noise
    price = price.clip(30000, None)

    df = pd.DataFrame({
        "Area_sqft": area.round().astype(int),
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Year_Built": year_built,
        "Location_Grade": location_grade,
        "Price": price.round(-2).astype(int),
    })
    return df


def load_data(path: str | None = None) -> pd.DataFrame:
    """Load data from CSV or generate a synthetic dataset."""
    if path:
        return pd.read_csv(path)
    return generate_dataset()


# ---------------------------------------------------------------------------
# 2. OUTLIER DETECTION & REMOVAL (IQR METHOD)
# ---------------------------------------------------------------------------

def remove_outliers_iqr(df: pd.DataFrame, columns: list[str] | None = None,
                        multiplier: float = 1.5) -> pd.DataFrame:
    """Remove rows with outliers in the specified columns using the IQR method."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR
        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
    return df_clean.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ---------------------------------------------------------------------------

def plot_price_distribution(df: pd.DataFrame, save_path: str | None = None) -> None:
    """Plot the distribution of the target variable (Price)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(df["Price"], kde=True, ax=axes[0], color="#2a9d8f", edgecolor="white")
    axes[0].set_title("Price Distribution", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Price ($)")
    axes[0].set_ylabel("Frequency")

    sns.boxplot(x=df["Price"], ax=axes[1], color="#e9c46a")
    axes[1].set_title("Price Box Plot (Outlier Detection)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Price ($)")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, save_path: str | None = None) -> None:
    """Plot correlation matrix heatmap for multicollinearity analysis."""
    corr = df.corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="YlGnBu",
                vmin=-1, vmax=1, linewidths=0.5, ax=ax, square=True)
    ax.set_title("Correlation Matrix — Multicollinearity Check", fontsize=14, fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# 4. MODEL TRAINING
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = ["Area_sqft", "Bedrooms", "Bathrooms", "Year_Built", "Location_Grade"]

# Global trained objects — populated by train_pipeline()
_trained_pipeline: Pipeline | None = None
_feature_columns: list[str] = FEATURE_COLUMNS


def train_pipeline(df: pd.DataFrame) -> tuple[Pipeline, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Build and train a GradientBoostingRegressor pipeline with StandardScaler.

    Returns:
        pipeline, X_test, y_test, y_pred, feature_columns
    """
    global _trained_pipeline

    X = df[_feature_columns]
    y = df["Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=4,
            subsample=0.9,
            random_state=42,
        )),
    ])

    pipeline.fit(X_train, y_train)
    _trained_pipeline = pipeline

    y_pred = pipeline.predict(X_test)
    return pipeline, X_test, y_test, y_pred, _feature_columns


# ---------------------------------------------------------------------------
# 5. MODEL EVALUATION
# ---------------------------------------------------------------------------

def evaluate_model(y_test: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Compute R², RMSE, and MAE evaluation metrics."""
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    metrics = {"R2": r2, "RMSE": rmse, "MAE": mae}

    print("=" * 50)
    print("MODEL EVALUATION METRICS")
    print("=" * 50)
    print(f"  R² Score   : {r2:.4f}")
    print(f"  RMSE       : ${rmse:,.2f}")
    print(f"  MAE        : ${mae:,.2f}")
    print("=" * 50)

    return metrics


# ---------------------------------------------------------------------------
# 6. VISUALIZATIONS
# ---------------------------------------------------------------------------

def plot_predicted_vs_actual(y_test: pd.Series, y_pred: np.ndarray,
                             save_path: str | None = None) -> None:
    """Scatter plot of Predicted vs Actual house prices."""
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(y_test, y_pred, alpha=0.5, color="#2a9d8f", edgecolors="white", s=40)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "--", color="#e76f51", linewidth=2, label="Perfect Prediction")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_title("Predicted vs Actual Prices", fontsize=14, fontweight="bold")
    ax.set_xlabel("Actual Price ($)")
    ax.set_ylabel("Predicted Price ($)")
    ax.legend(fontsize=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_feature_importance(pipeline: Pipeline, feature_names: list[str],
                            save_path: str | None = None) -> None:
    """Bar chart of feature importances from the trained model."""
    model = pipeline.named_steps["model"]
    importances = model.feature_importances_

    order = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in order]
    sorted_values = importances[order]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(sorted_names[::-1], sorted_values[::-1],
                   color=["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"])

    ax.set_title("Feature Importance", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance Score")

    for bar, val in zip(bars, sorted_values[::-1]):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# 7. PRODUCTION PREDICTION FUNCTION (AUTOGRADER ENTRY POINT)
# ---------------------------------------------------------------------------

def predict_house_price(area: float, bedrooms: int, bathrooms: int,
                        year_built: int, location_grade: int) -> float:
    """Predict house price from raw numeric inputs.

    This is the standalone production function required for the GitHub
    Autograder.  It uses the globally trained pipeline; if the pipeline
    has not been trained yet it will train on a fresh synthetic dataset.

    Parameters
    ----------
    area : float          — Living area in square feet
    bedrooms : int        — Number of bedrooms
    bathrooms : int       — Number of bathrooms
    year_built : int      — Year the house was built
    location_grade : int  — Location quality grade (1–5)

    Returns
    -------
    float — Predicted house price
    """
    global _trained_pipeline

    if _trained_pipeline is None:
        print("[INFO] Pipeline not trained yet — training on synthetic data...")
        df = generate_dataset()
        df_clean = remove_outliers_iqr(df)
        train_pipeline(df_clean)

    input_df = pd.DataFrame([{
        "Area_sqft": float(area),
        "Bedrooms": int(bedrooms),
        "Bathrooms": int(bathrooms),
        "Year_Built": int(year_built),
        "Location_Grade": int(location_grade),
    }])

    prediction = _trained_pipeline.predict(input_df)
    return float(prediction[0])


# ---------------------------------------------------------------------------
# MAIN EXECUTION (JUPYTER NOTEBOOK COMPATIBLE)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Load / Generate Data ---
    df = load_data()
    print(f"Raw dataset shape: {df.shape}")
    print(df.head())
    print(df.describe())

    # --- Outlier Removal ---
    df_clean = remove_outliers_iqr(df)
    print(f"\nAfter IQR outlier removal: {df_clean.shape}  "
          f"(removed {df.shape[0] - df_clean.shape[0]} rows)")

    # --- EDA ---
    plot_price_distribution(df_clean, save_path="price_distribution.png")
    plot_correlation_heatmap(df_clean, save_path="correlation_heatmap.png")

    # --- Train Model ---
    pipeline, X_test, y_test, y_pred, feat_cols = train_pipeline(df_clean)

    # --- Evaluate ---
    metrics = evaluate_model(y_test, y_pred)

    # --- Visualize Results ---
    plot_predicted_vs_actual(y_test, y_pred, save_path="predicted_vs_actual.png")
    plot_feature_importance(pipeline, feat_cols, save_path="feature_importance.png")

    # --- Test Production Function ---
    sample_price = predict_house_price(
        area=2000, bedrooms=3, bathrooms=2, year_built=2015, location_grade=4
    )
    print(f"\nPredicted price for sample house: ${sample_price:,.2f}")
