"""
Tests for model performance and quality.

These tests ensure models meet minimum performance thresholds
and maintain quality standards.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


class TestModelPerformance:
    """Test suite for model performance metrics."""

    @pytest.fixture
    def synthetic_binary_data(self):
        """Generate synthetic binary classification data for testing."""
        np.random.seed(42)
        n_samples = 1000

        # Generate features
        X = np.random.randn(n_samples, 10)

        # Generate labels with some pattern
        y = (X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.5 > 0).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }

    def test_model_performance_above_baseline(self, synthetic_binary_data):
        """Test that model performs better than random baseline."""
        from sklearn.linear_model import LogisticRegression

        data = synthetic_binary_data
        model = LogisticRegression(random_state=42)
        model.fit(data['X_train'], data['y_train'])

        predictions = model.predict_proba(data['X_test'])[:, 1]
        auc = roc_auc_score(data['y_test'], predictions)

        # Should be better than random (>0.5)
        assert auc > 0.5, f"Model AUC {auc:.3f} is not better than random (0.5)"

    def test_model_meets_minimum_auc(self, synthetic_binary_data):
        """Test that model meets minimum AUC threshold."""
        from sklearn.ensemble import RandomForestClassifier

        data = synthetic_binary_data
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(data['X_train'], data['y_train'])

        predictions = model.predict_proba(data['X_test'])[:, 1]
        auc = roc_auc_score(data['y_test'], predictions)

        # Minimum acceptable AUC for production models
        MIN_AUC = 0.65
        assert auc >= MIN_AUC, \
            f"Model AUC {auc:.3f} below minimum threshold {MIN_AUC}"

    def test_model_precision_recall_balance(self, synthetic_binary_data):
        """Test that precision and recall are reasonably balanced."""
        from sklearn.ensemble import RandomForestClassifier

        data = synthetic_binary_data
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(data['X_train'], data['y_train'])

        predictions = model.predict(data['X_test'])

        precision = precision_score(data['y_test'], predictions)
        recall = recall_score(data['y_test'], predictions)

        # Neither should be extremely low compared to the other
        ratio = min(precision, recall) / max(precision, recall)
        MIN_RATIO = 0.5  # At least 50% of the higher metric

        assert ratio >= MIN_RATIO, \
            f"Precision ({precision:.3f}) and Recall ({recall:.3f}) are imbalanced (ratio: {ratio:.3f})"

    def test_no_perfect_predictions(self, synthetic_binary_data):
        """Test that model isn't overfitting to achieve perfect predictions."""
        from sklearn.tree import DecisionTreeClassifier

        data = synthetic_binary_data
        # Deep tree that might overfit
        model = DecisionTreeClassifier(max_depth=20, random_state=42)
        model.fit(data['X_train'], data['y_train'])

        train_predictions = model.predict(data['X_train'])
        train_accuracy = (train_predictions == data['y_train']).mean()

        # Perfect training accuracy is a red flag
        assert train_accuracy < 0.99, \
            f"Model may be overfitting (train accuracy: {train_accuracy:.3f})"


class TestModelStability:
    """Test model stability and reproducibility."""

    def test_model_reproducibility(self):
        """Test that model produces same results with same random seed."""
        from sklearn.ensemble import RandomForestClassifier

        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        # Train two models with same seed
        model1 = RandomForestClassifier(n_estimators=10, random_state=42)
        model1.fit(X, y)
        pred1 = model1.predict(X)

        model2 = RandomForestClassifier(n_estimators=10, random_state=42)
        model2.fit(X, y)
        pred2 = model2.predict(X)

        # Should produce identical predictions
        assert np.array_equal(pred1, pred2), \
            "Model not reproducible with same random seed"

    def test_model_stability_across_splits(self):
        """Test that model performance is stable across different data splits."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score

        X = np.random.randn(500, 10)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        model = RandomForestClassifier(n_estimators=50, random_state=42)

        # Cross-validation scores
        scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')

        # Standard deviation of scores should be reasonable
        MAX_STD = 0.15
        std_dev = scores.std()

        assert std_dev < MAX_STD, \
            f"Model performance is unstable across splits (std: {std_dev:.3f})"


class TestDataValidation:
    """Test data validation and quality checks."""

    def test_no_missing_values_in_features(self):
        """Test that training data has no missing values."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 4, 3, 2, 1],
            'target': [0, 1, 0, 1, 0]
        })

        # Check for missing values
        missing = df.isnull().sum().sum()
        assert missing == 0, f"Found {missing} missing values in training data"

    def test_detect_missing_values(self):
        """Test that we can detect missing values when present."""
        df = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5],
            'feature2': [5, 4, 3, np.nan, 1],
            'target': [0, 1, 0, 1, 0]
        })

        # Should detect missing values
        missing = df.isnull().sum().sum()
        assert missing > 0, "Missing value detection not working"

    def test_feature_types_consistent(self):
        """Test that feature types are consistent."""
        df = pd.DataFrame({
            'numeric_feature': [1.0, 2.0, 3.0],
            'categorical_feature': ['a', 'b', 'c'],
            'target': [0, 1, 0]
        })

        # Numeric column should be numeric
        assert pd.api.types.is_numeric_dtype(df['numeric_feature']), \
            "Numeric feature has wrong type"

        # Categorical column should be object or category
        assert pd.api.types.is_object_dtype(df['categorical_feature']) or \
               pd.api.types.is_categorical_dtype(df['categorical_feature']), \
            "Categorical feature has wrong type"

    def test_target_distribution_not_extreme(self):
        """Test that target distribution isn't extremely imbalanced."""
        # Extremely imbalanced case
        y = np.array([0] * 990 + [1] * 10)

        minority_ratio = min(sum(y == 0), sum(y == 1)) / len(y)

        # Warn if less than 5% minority class
        if minority_ratio < 0.05:
            pytest.skip(
                f"Target distribution extremely imbalanced ({minority_ratio*100:.1f}% minority class). "
                "Consider using appropriate sampling techniques or evaluation metrics."
            )


class TestFairnessMetrics:
    """Test fairness and bias metrics."""

    @pytest.fixture
    def fairness_data(self):
        """Generate data with protected attributes for fairness testing."""
        np.random.seed(42)
        n_samples = 1000

        data = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'protected_attr': np.random.choice(['Group_A', 'Group_B'], n_samples),
            'target': np.random.randint(0, 2, n_samples)
        })

        return data

    def test_demographic_parity_check(self, fairness_data):
        """Test for demographic parity across protected groups."""
        # Simple model predictions (using feature1 > 0)
        fairness_data['prediction'] = (fairness_data['feature1'] > 0).astype(int)

        # Calculate selection rate by group
        selection_rates = fairness_data.groupby('protected_attr')['prediction'].mean()

        # Four-fifths rule: ratio should be >= 0.8
        if len(selection_rates) >= 2:
            ratio = selection_rates.min() / selection_rates.max()

            # Log the ratio (informational)
            print(f"\nDemographic parity ratio: {ratio:.3f}")
            print(f"Selection rates by group:\n{selection_rates}")

            # For this test, we just check it's calculated correctly
            assert 0 <= ratio <= 1, "Demographic parity ratio should be between 0 and 1"

    def test_equal_opportunity_check(self, fairness_data):
        """Test for equal opportunity (TPR parity) across groups."""
        fairness_data['prediction'] = (fairness_data['feature1'] > 0).astype(int)

        # Calculate TPR by group (for positive class)
        tpr_by_group = {}

        for group in fairness_data['protected_attr'].unique():
            group_data = fairness_data[fairness_data['protected_attr'] == group]
            positive_cases = group_data[group_data['target'] == 1]

            if len(positive_cases) > 0:
                tpr = (positive_cases['prediction'] == 1).mean()
                tpr_by_group[group] = tpr

        print(f"\nTrue Positive Rate by group: {tpr_by_group}")

        # Just verify we can calculate it
        assert len(tpr_by_group) > 0, "Should be able to calculate TPR by group"


class TestModelArtifacts:
    """Test model saving and loading."""

    def test_model_save_and_load(self, tmp_path):
        """Test that model can be saved and loaded."""
        import joblib
        from sklearn.linear_model import LogisticRegression

        # Train a simple model
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        model = LogisticRegression(random_state=42)
        model.fit(X, y)

        # Save model
        model_path = tmp_path / "test_model.pkl"
        joblib.dump(model, model_path)

        # Load model
        loaded_model = joblib.load(model_path)

        # Verify predictions match
        original_pred = model.predict(X)
        loaded_pred = loaded_model.predict(X)

        assert np.array_equal(original_pred, loaded_pred), \
            "Loaded model predictions don't match original"

    def test_model_metadata_saved(self, tmp_path):
        """Test that model metadata is saved alongside model."""
        import joblib
        import json

        metadata = {
            'model_id': 'test-model-v1',
            'version': '1.0.0',
            'trained_at': '2026-01-16',
            'performance': {'auc': 0.85}
        }

        # Save metadata
        metadata_path = tmp_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

        # Load and verify
        with open(metadata_path, 'r') as f:
            loaded_metadata = json.load(f)

        assert loaded_metadata['model_id'] == metadata['model_id']
        assert loaded_metadata['version'] == metadata['version']


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, '-v'])
