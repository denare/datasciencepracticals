# =============================================================================
# DEEP DIVE 5: Scikit-learn Complete Reference 🤖
# =============================================================================
# Scikit-learn is the standard ML library in Python. It provides:
# - Preprocessing & Feature Engineering
# - Classification, Regression, Clustering
# - Model Selection & Evaluation
# - Pipelines
#
# This script covers EVERYTHING you need:
#   Part A: Preprocessing (Scaling, Encoding, Imputation)
#   Part B: Classification Models (LogReg, KNN, SVM, Trees, Ensemble)
#   Part C: Regression Models (Linear, Ridge, Lasso, Decision Tree)
#   Part D: Model Evaluation Metrics
#   Part E: Cross-Validation
#   Part F: Hyperparameter Tuning (GridSearch, RandomSearch)
#   Part G: Pipelines
#   Part H: Clustering (KMeans, DBSCAN)
#   Part I: Dimensionality Reduction (PCA, t-SNE)
#   Part J: Feature Selection
#   Part K: Full Pipeline Example
# =============================================================================

import os
import warnings
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


def part_a_preprocessing():
    """Part A: Preprocessing — Scaling, Encoding, Imputation."""
    print("=" * 60)
    print("PART A: Preprocessing")
    print("=" * 60)

    from sklearn.preprocessing import (
        StandardScaler, MinMaxScaler, LabelEncoder,
        OneHotEncoder, OrdinalEncoder
    )
    from sklearn.impute import SimpleImputer

    # --- Scaling ---
    data = np.array([[100, 0.1], [200, 0.2], [300, 0.3], [400, 0.4]])

    # StandardScaler: mean=0, std=1 (z-score normalization)
    scaler_std = StandardScaler()
    scaled_std = scaler_std.fit_transform(data)
    print("StandardScaler (z-score):\n", scaled_std)

    # MinMaxScaler: scales to [0, 1]
    scaler_mm = MinMaxScaler()
    scaled_mm = scaler_mm.fit_transform(data)
    print("\nMinMaxScaler [0,1]:\n", scaled_mm)

    # --- Encoding ---
    # LabelEncoder: text → integers (for target variable)
    le = LabelEncoder()
    labels = ['cat', 'dog', 'bird', 'cat', 'dog']
    encoded = le.fit_transform(labels)
    print("\nLabelEncoder:", labels, "→", encoded)
    print("Inverse:", le.inverse_transform(encoded))

    # OneHotEncoder: categorical → binary columns
    ohe = OneHotEncoder(sparse_output=False)
    categories = np.array([['red'], ['blue'], ['green'], ['red']])
    one_hot = ohe.fit_transform(categories)
    print("\nOneHotEncoder:\n", one_hot)
    print("Categories:", ohe.categories_)

    # OrdinalEncoder: ordered categories
    oe = OrdinalEncoder(categories=[['small', 'medium', 'large']])
    sizes = np.array([['medium'], ['large'], ['small'], ['large']])
    ordinal = oe.fit_transform(sizes)
    print("\nOrdinalEncoder:", sizes.flatten(), "→", ordinal.flatten())

    # --- Imputation ---
    data_with_nan = np.array([[1, np.nan], [2, 3], [np.nan, 4], [5, 6]])
    imputer = SimpleImputer(strategy='mean')  # Also: 'median', 'most_frequent', 'constant'
    imputed = imputer.fit_transform(data_with_nan)
    print("\nSimpleImputer (mean):\n", imputed, "\n")


def part_b_classification():
    """Part B: Classification Models."""
    print("=" * 60)
    print("PART B: Classification Models")
    print("=" * 60)

    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Classifiers
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import (
        RandomForestClassifier, GradientBoostingClassifier,
        AdaBoostClassifier, BaggingClassifier
    )

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Support Vector Machine': SVC(kernel='rbf'),
        'Decision Tree': DecisionTreeClassifier(max_depth=5),
        'Random Forest': RandomForestClassifier(n_estimators=100),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100),
        'AdaBoost': AdaBoostClassifier(n_estimators=50),
        'Bagging': BaggingClassifier(n_estimators=50),
    }

    print(f"{'Model':<25} {'Accuracy':>10}")
    print("-" * 37)
    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"  {name:<25} {acc*100:>8.2f}%")
    print()


def part_c_regression():
    """Part C: Regression Models."""
    print("=" * 60)
    print("PART C: Regression Models")
    print("=" * 60)

    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score

    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

    data = fetch_california_housing()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    models = {
        'Linear Regression': LinearRegression(),
        'Ridge (L2)': Ridge(alpha=1.0),
        'Lasso (L1)': Lasso(alpha=0.01),
        'ElasticNet (L1+L2)': ElasticNet(alpha=0.01),
        'Decision Tree': DecisionTreeRegressor(max_depth=10),
        'Random Forest': RandomForestRegressor(n_estimators=50),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100),
    }

    print(f"{'Model':<25} {'RMSE':>8} {'R²':>8}")
    print("-" * 43)
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)
        print(f"  {name:<25} {rmse:>8.4f} {r2:>8.4f}")
    print()


def part_d_metrics():
    """Part D: Model Evaluation Metrics."""
    print("=" * 60)
    print("PART D: Model Evaluation Metrics")
    print("=" * 60)

    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report, roc_auc_score,
        mean_absolute_error, mean_squared_error, r2_score
    )

    # Classification metrics
    y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
    y_pred = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]

    print("--- Classification Metrics ---")
    print(f"  Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"  Recall   : {recall_score(y_true, y_pred):.4f}")
    print(f"  F1 Score : {f1_score(y_true, y_pred):.4f}")
    print(f"  ROC AUC  : {roc_auc_score(y_true, y_pred):.4f}")
    print(f"\n  Confusion Matrix:\n  {confusion_matrix(y_true, y_pred)}")
    print(f"\n  Classification Report:\n{classification_report(y_true, y_pred)}")

    # Regression metrics
    y_true_r = [3.0, -0.5, 2.0, 7.0]
    y_pred_r = [2.5, 0.0, 2.1, 7.8]

    print("--- Regression Metrics ---")
    print(f"  MAE  : {mean_absolute_error(y_true_r, y_pred_r):.4f}")
    print(f"  RMSE : {np.sqrt(mean_squared_error(y_true_r, y_pred_r)):.4f}")
    print(f"  R²   : {r2_score(y_true_r, y_pred_r):.4f}\n")


def part_e_cross_validation():
    """Part E: Cross-Validation."""
    print("=" * 60)
    print("PART E: Cross-Validation")
    print("=" * 60)

    from sklearn.datasets import load_iris
    from sklearn.model_selection import (
        cross_val_score, KFold, StratifiedKFold, LeaveOneOut
    )
    from sklearn.ensemble import RandomForestClassifier

    X, y = load_iris(return_X_y=True)
    model = RandomForestClassifier(n_estimators=50, random_state=42)

    # Basic cross-validation
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"5-Fold CV scores: {scores}")
    print(f"Mean: {scores.mean():.4f} ± {scores.std():.4f}")

    # KFold (manual control)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores_kf = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    print(f"\nKFold (shuffled): {scores_kf.mean():.4f}")

    # StratifiedKFold (preserves class ratios — best for imbalanced data)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores_skf = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    print(f"StratifiedKFold: {scores_skf.mean():.4f}\n")


def part_f_tuning():
    """Part F: Hyperparameter Tuning."""
    print("=" * 60)
    print("PART F: Hyperparameter Tuning")
    print("=" * 60)

    from sklearn.datasets import load_iris
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    from sklearn.ensemble import RandomForestClassifier

    X, y = load_iris(return_X_y=True)

    # GridSearchCV — tries EVERY combination
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5, None],
        'min_samples_split': [2, 5]
    }
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid, cv=3, scoring='accuracy', n_jobs=-1
    )
    grid.fit(X, y)
    print("GridSearchCV:")
    print(f"  Best params: {grid.best_params_}")
    print(f"  Best score : {grid.best_score_:.4f}")

    # RandomizedSearchCV — samples random combinations (faster)
    from scipy.stats import randint
    param_dist = {
        'n_estimators': randint(50, 200),
        'max_depth': [3, 5, 10, None],
        'min_samples_split': randint(2, 10)
    }
    random_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_dist, n_iter=10, cv=3, scoring='accuracy', random_state=42
    )
    random_search.fit(X, y)
    print(f"\nRandomizedSearchCV:")
    print(f"  Best params: {random_search.best_params_}")
    print(f"  Best score : {random_search.best_score_:.4f}\n")


def part_g_pipelines():
    """Part G: Pipelines — Chain preprocessing + model."""
    print("=" * 60)
    print("PART G: Pipelines")
    print("=" * 60)

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import load_iris
    from sklearn.model_selection import cross_val_score

    X, y = load_iris(return_X_y=True)

    # Simple pipeline: Scale → PCA → Classify
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=2)),
        ('classifier', LogisticRegression(max_iter=1000))
    ])

    scores = cross_val_score(pipe, X, y, cv=5, scoring='accuracy')
    print("Pipeline (Scale → PCA → LogReg):")
    print(f"  CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

    # ColumnTransformer — different transforms for different columns
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder

    print("\n  ColumnTransformer lets you apply different transforms")
    print("  to numeric vs. categorical columns in ONE step.")
    print("  Example:")
    print("    numeric_features   → StandardScaler")
    print("    categorical_features → OneHotEncoder\n")


def part_h_clustering():
    """Part H: Clustering."""
    print("=" * 60)
    print("PART H: Clustering")
    print("=" * 60)
    os.makedirs('plots/deep_dive', exist_ok=True)

    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.datasets import make_blobs
    from sklearn.metrics import silhouette_score

    # Generate sample data
    X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # KMeans
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels_km = kmeans.fit_predict(X)
    axes[0].scatter(X[:, 0], X[:, 1], c=labels_km, cmap='viridis', s=20)
    axes[0].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                    c='red', marker='X', s=200, label='Centers')
    axes[0].set_title(f'KMeans (silhouette: {silhouette_score(X, labels_km):.2f})')
    axes[0].legend()

    # DBSCAN
    dbscan = DBSCAN(eps=0.8, min_samples=5)
    labels_db = dbscan.fit_predict(X)
    axes[1].scatter(X[:, 0], X[:, 1], c=labels_db, cmap='viridis', s=20)
    axes[1].set_title(f'DBSCAN ({len(set(labels_db)) - 1} clusters)')

    # Agglomerative
    agg = AgglomerativeClustering(n_clusters=4)
    labels_agg = agg.fit_predict(X)
    axes[2].scatter(X[:, 0], X[:, 1], c=labels_agg, cmap='viridis', s=20)
    axes[2].set_title(f'Agglomerative (silhouette: {silhouette_score(X, labels_agg):.2f})')

    plt.tight_layout()
    plt.savefig('plots/deep_dive/sklearn_clustering.png')
    plt.close()
    print("Saved 'plots/deep_dive/sklearn_clustering.png'\n")


def part_i_dimensionality():
    """Part I: Dimensionality Reduction."""
    print("=" * 60)
    print("PART I: Dimensionality Reduction")
    print("=" * 60)

    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # PCA — Principal Component Analysis
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=30)
    axes[0].set_title(f'PCA (explained var: {pca.explained_variance_ratio_.sum():.2%})')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')

    # t-SNE — non-linear dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X)
    axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', s=30)
    axes[1].set_title('t-SNE')
    axes[1].set_xlabel('Dim 1')
    axes[1].set_ylabel('Dim 2')

    plt.tight_layout()
    plt.savefig('plots/deep_dive/sklearn_dim_reduction.png')
    plt.close()
    print("Saved 'plots/deep_dive/sklearn_dim_reduction.png'\n")


def part_j_feature_selection():
    """Part J: Feature Selection."""
    print("=" * 60)
    print("PART J: Feature Selection")
    print("=" * 60)

    from sklearn.datasets import load_iris
    from sklearn.feature_selection import (
        SelectKBest, f_classif, mutual_info_classif
    )
    from sklearn.ensemble import RandomForestClassifier

    X, y = load_iris(return_X_y=True)
    feature_names = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid']

    # Method 1: SelectKBest (statistical test)
    selector = SelectKBest(f_classif, k=2)
    X_selected = selector.fit_transform(X, y)
    selected = [feature_names[i] for i in selector.get_support(indices=True)]
    print(f"SelectKBest (top 2): {selected}")

    # Method 2: Feature importance from tree models
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
    importance = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
    print(f"\nRandom Forest importances:\n{importance}\n")


def part_k_full_pipeline():
    """Part K: Full Pipeline Example — End-to-end workflow."""
    print("=" * 60)
    print("PART K: Full Pipeline Example")
    print("=" * 60)

    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score

    # Load Titanic
    df = sns.load_dataset('titanic').dropna(subset=['embarked'])
    df = df[['survived', 'pclass', 'sex', 'age', 'fare', 'embarked']].copy()
    df['age'] = df['age'].fillna(df['age'].median())

    X = df.drop('survived', axis=1)
    y = df['survived']

    # Define transforms per column type
    numeric_features = ['age', 'fare', 'pclass']
    categorical_features = ['sex', 'embarked']

    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

    # Full pipeline: preprocess → model
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))
    ])

    # Evaluate with cross-validation
    scores = cross_val_score(full_pipeline, X, y, cv=5, scoring='accuracy')
    print(f"Full Pipeline CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
    print("  Steps: Impute → Scale/Encode → GradientBoosting\n")


def main():
    part_a_preprocessing()
    part_b_classification()
    part_c_regression()
    part_d_metrics()
    part_e_cross_validation()
    part_f_tuning()
    part_g_pipelines()
    part_h_clustering()
    part_i_dimensionality()
    part_j_feature_selection()
    part_k_full_pipeline()

    print("=" * 60)
    print("🎉 Scikit-learn Deep Dive Complete!")
    print("You now know the full Scikit-learn toolkit for ML.")
    print("=" * 60)

if __name__ == "__main__":
    main()
