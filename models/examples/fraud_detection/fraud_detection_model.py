"""
Fraud Detection Model - Example Implementation

This is a simple fraud detection model for demonstration purposes.
It shows how to structure an ML model with proper governance practices.

Risk Tier: Medium
Model ID: fraud-detection-v1
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
import json
import logging
from datetime import datetime
from typing import Dict, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FraudDetectionModel:
    """
    Random Forest based fraud detection model.

    This model predicts the likelihood of fraudulent transactions based on
    transaction features such as amount, time, location, and user behavior.
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize the fraud detection model.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.metadata = {
            'model_id': 'fraud-detection-v1',
            'model_version': '1.0.0',
            'risk_tier': 'medium',
            'created_at': None,
            'trained_at': None,
            'performance_metrics': {}
        }

        # Set random seeds for reproducibility
        np.random.seed(self.random_state)

        logger.info(f"Initialized FraudDetectionModel with random_state={random_state}")

    def create_model(self) -> RandomForestClassifier:
        """
        Create and configure the Random Forest model.

        Returns:
            Configured RandomForestClassifier instance
        """
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            random_state=self.random_state,
            n_jobs=-1,
            class_weight='balanced'  # Handle class imbalance
        )
        logger.info("Created RandomForestClassifier with configured hyperparameters")
        return model

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare and engineer features from raw transaction data.

        Args:
            df: Raw transaction dataframe

        Returns:
            DataFrame with engineered features
        """
        features = df.copy()

        # Log transformation for amount (handle large values)
        features['log_amount'] = np.log1p(features['amount'])

        # Time-based features
        features['hour_of_day'] = features['timestamp'].dt.hour
        features['day_of_week'] = features['timestamp'].dt.dayofweek
        features['is_weekend'] = features['day_of_week'].isin([5, 6]).astype(int)

        # User behavior features
        features['transactions_last_24h'] = features.groupby('user_id')['timestamp'].transform(
            lambda x: x.rolling('24h').count()
        )

        logger.info(f"Engineered {len(features.columns)} features from raw data")
        return features

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None
    ) -> Dict[str, float]:
        """
        Train the fraud detection model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Returns:
            Dictionary of performance metrics
        """
        logger.info(f"Starting training with {len(X_train)} samples")

        # Store feature names
        self.feature_names = X_train.columns.tolist()

        # Fit scaler on training data
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Create and train model
        self.model = self.create_model()
        self.model.fit(X_train_scaled, y_train)

        # Record training timestamp
        self.metadata['trained_at'] = datetime.now().isoformat()
        self.metadata['training_samples'] = len(X_train)
        self.metadata['feature_count'] = len(self.feature_names)

        # Evaluate on training set
        train_predictions = self.model.predict(X_train_scaled)
        train_proba = self.model.predict_proba(X_train_scaled)[:, 1]

        metrics = {
            'train_auc': float(roc_auc_score(y_train, train_proba))
        }

        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            val_predictions = self.model.predict(X_val_scaled)
            val_proba = self.model.predict_proba(X_val_scaled)[:, 1]

            metrics['val_auc'] = float(roc_auc_score(y_val, val_proba))

            # Generate classification report
            report = classification_report(y_val, val_predictions, output_dict=True)
            metrics['val_precision'] = float(report['1']['precision'])
            metrics['val_recall'] = float(report['1']['recall'])
            metrics['val_f1'] = float(report['1']['f1-score'])

            logger.info(f"Validation AUC: {metrics['val_auc']:.4f}")

        # Store metrics
        self.metadata['performance_metrics'] = metrics

        # Calculate feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        logger.info(f"Top 5 features: {feature_importance.head().to_dict('records')}")

        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict fraud labels for transactions.

        Args:
            X: Feature dataframe

        Returns:
            Array of predictions (0 = legitimate, 1 = fraud)
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)

        logger.info(f"Generated predictions for {len(X)} samples")
        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict fraud probabilities for transactions.

        Args:
            X: Feature dataframe

        Returns:
            Array of fraud probabilities
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        X_scaled = self.scaler.transform(X)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]

        return probabilities

    def save(self, model_path: str, metadata_path: str = None):
        """
        Save the trained model and metadata.

        Args:
            model_path: Path to save model artifacts
            metadata_path: Path to save metadata (optional)
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        # Save model and scaler
        artifacts = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'metadata': self.metadata
        }

        joblib.dump(artifacts, model_path)
        logger.info(f"Model saved to {model_path}")

        # Save metadata separately if requested
        if metadata_path:
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            logger.info(f"Metadata saved to {metadata_path}")

    def load(self, model_path: str):
        """
        Load a trained model from disk.

        Args:
            model_path: Path to model artifacts
        """
        artifacts = joblib.load(model_path)

        self.model = artifacts['model']
        self.scaler = artifacts['scaler']
        self.feature_names = artifacts['feature_names']
        self.metadata = artifacts['metadata']

        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Model version: {self.metadata.get('model_version')}")


def generate_synthetic_data(n_samples: int = 10000, fraud_rate: float = 0.02) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generate synthetic transaction data for demonstration.

    Args:
        n_samples: Number of samples to generate
        fraud_rate: Proportion of fraudulent transactions

    Returns:
        Tuple of (features, labels)
    """
    np.random.seed(42)

    # Generate timestamps
    base_time = pd.Timestamp('2025-01-01')
    timestamps = [base_time + pd.Timedelta(minutes=i) for i in range(n_samples)]

    # Generate features
    data = {
        'timestamp': timestamps,
        'user_id': np.random.randint(1, 1000, n_samples),
        'amount': np.random.lognormal(4, 1.5, n_samples),  # Log-normal distribution for amounts
        'merchant_category': np.random.choice(['retail', 'online', 'grocery', 'gas', 'restaurant'], n_samples),
        'device_type': np.random.choice(['mobile', 'desktop', 'tablet'], n_samples),
        'location_change': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
    }

    df = pd.DataFrame(data)

    # Generate labels (0 = legitimate, 1 = fraud)
    n_fraud = int(n_samples * fraud_rate)
    labels = np.array([0] * (n_samples - n_fraud) + [1] * n_fraud)
    np.random.shuffle(labels)

    # Make fraudulent transactions have different patterns
    fraud_mask = labels == 1
    df.loc[fraud_mask, 'amount'] *= 2  # Larger amounts
    df.loc[fraud_mask, 'location_change'] = 1  # More location changes

    logger.info(f"Generated {n_samples} synthetic transactions ({fraud_rate*100}% fraud)")

    return df, pd.Series(labels)


if __name__ == "__main__":
    """
    Example usage of the fraud detection model.
    """

    # Generate synthetic data
    print("Generating synthetic transaction data...")
    data, labels = generate_synthetic_data(n_samples=10000, fraud_rate=0.02)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42, stratify=labels
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # Initialize model
    print("\nInitializing fraud detection model...")
    model = FraudDetectionModel(random_state=42)

    # Prepare features
    print("Engineering features...")
    X_train_features = model.prepare_features(X_train)
    X_val_features = model.prepare_features(X_val)
    X_test_features = model.prepare_features(X_test)

    # Select numerical features for training
    feature_cols = ['log_amount', 'hour_of_day', 'day_of_week', 'is_weekend', 'location_change']

    # Train model
    print("\nTraining model...")
    metrics = model.train(
        X_train_features[feature_cols],
        y_train,
        X_val_features[feature_cols],
        y_val
    )

    print(f"\nTraining Results:")
    print(f"  Train AUC: {metrics['train_auc']:.4f}")
    print(f"  Val AUC: {metrics['val_auc']:.4f}")
    print(f"  Val Precision: {metrics['val_precision']:.4f}")
    print(f"  Val Recall: {metrics['val_recall']:.4f}")
    print(f"  Val F1: {metrics['val_f1']:.4f}")

    # Test model
    print("\nEvaluating on test set...")
    test_predictions = model.predict(X_test_features[feature_cols])
    test_proba = model.predict_proba(X_test_features[feature_cols])

    test_auc = roc_auc_score(y_test, test_proba)
    print(f"  Test AUC: {test_auc:.4f}")

    # Show confusion matrix
    cm = confusion_matrix(y_test, test_predictions)
    print("\nConfusion Matrix:")
    print(f"  TN: {cm[0,0]:4d}  FP: {cm[0,1]:4d}")
    print(f"  FN: {cm[1,0]:4d}  TP: {cm[1,1]:4d}")

    # Save model
    print("\nSaving model...")
    model.save('fraud_detection_model.pkl', 'fraud_detection_metadata.json')

    print("\nModel training complete!")
