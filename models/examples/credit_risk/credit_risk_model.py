"""
Credit Risk Assessment Model - Example Implementation

This is a credit risk assessment model for demonstration purposes.
It shows how to structure a HIGH-RISK ML model with comprehensive governance practices.

Risk Tier: High
Model ID: credit-risk-v2
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix
)
import joblib
import json
import logging
from datetime import datetime
from typing import Dict, Any, Tuple
import warnings

warnings.filterwarnings('ignore')

# Configure logging with audit trail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CreditRiskModel:
    """
    Logistic Regression based credit risk assessment model.

    This model predicts the probability of loan default based on applicant
    financial and demographic features. Given the HIGH RISK tier, this model
    includes enhanced governance, monitoring, and fairness capabilities.
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize the credit risk model.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.protected_attributes = ['age_group', 'gender']  # For fairness monitoring
        self.decision_threshold = 0.5  # Default threshold for approval

        self.metadata = {
            'model_id': 'credit-risk-v2',
            'model_version': '2.0.0',
            'risk_tier': 'high',
            'created_at': datetime.now().isoformat(),
            'trained_at': None,
            'performance_metrics': {},
            'fairness_metrics': {},
            'regulatory_compliance': {
                'sr_11_7': True,
                'equal_credit_opportunity_act': True
            },
            'approval_required': {
                'validator': None,
                'compliance_officer': None,
                'business_owner': None
            }
        }

        # Set random seeds for reproducibility
        np.random.seed(self.random_state)

        logger.info(f"Initialized CreditRiskModel (HIGH RISK) with random_state={random_state}")

    def create_model(self) -> LogisticRegression:
        """
        Create and configure the Logistic Regression model.

        We use Logistic Regression for interpretability - critical for
        high-risk financial models where explanations are required.

        Returns:
            Configured LogisticRegression instance
        """
        model = LogisticRegression(
            penalty='l2',
            C=1.0,
            solver='lbfgs',
            max_iter=1000,
            random_state=self.random_state,
            class_weight='balanced'  # Handle class imbalance
        )
        logger.info("Created LogisticRegression model (chosen for interpretability)")
        return model

    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate input data quality - critical for high-risk models.

        Args:
            df: Input dataframe

        Returns:
            Dictionary of validation results
        """
        validation_results = {
            'passed': True,
            'issues': []
        }

        # Check for missing values
        missing_pct = (df.isnull().sum() / len(df) * 100).to_dict()
        for col, pct in missing_pct.items():
            if pct > 5:  # Flag if >5% missing
                validation_results['issues'].append(
                    f"Column '{col}' has {pct:.1f}% missing values"
                )
                validation_results['passed'] = False

        # Check for duplicate records
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            validation_results['issues'].append(
                f"Found {duplicates} duplicate records"
            )
            logger.warning(f"Data quality issue: {duplicates} duplicates found")

        # Check value ranges
        if 'annual_income' in df.columns:
            if (df['annual_income'] < 0).any():
                validation_results['issues'].append("Negative income values detected")
                validation_results['passed'] = False

        logger.info(f"Data quality validation: {'PASSED' if validation_results['passed'] else 'FAILED'}")
        return validation_results

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare and engineer features from applicant data.

        Args:
            df: Raw applicant dataframe

        Returns:
            DataFrame with engineered features
        """
        features = df.copy()

        # Debt-to-income ratio (key financial indicator)
        features['debt_to_income'] = features['monthly_debt'] / (features['annual_income'] / 12)
        features['debt_to_income'] = features['debt_to_income'].clip(upper=2.0)  # Cap outliers

        # Credit utilization
        features['credit_utilization'] = features['credit_used'] / features['credit_limit'].replace(0, 1)
        features['credit_utilization'] = features['credit_utilization'].clip(upper=1.5)

        # Income stability indicator
        features['income_stability'] = features['years_employed'] / (features['age'] - 18).replace(0, 1)

        # Loan-to-income ratio
        features['loan_to_income'] = features['loan_amount'] / features['annual_income'].replace(0, 1)

        # Age groups for fairness analysis (but not used as model feature)
        features['age_group'] = pd.cut(
            features['age'],
            bins=[0, 25, 35, 50, 100],
            labels=['18-25', '26-35', '36-50', '50+']
        )

        logger.info(f"Engineered {len(features.columns)} features")
        logger.info(f"Protected attributes tracked: {self.protected_attributes}")

        return features

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        protected_train: pd.DataFrame = None,
        protected_val: pd.DataFrame = None
    ) -> Dict[str, float]:
        """
        Train the credit risk model with comprehensive validation.

        Args:
            X_train: Training features
            y_train: Training labels (1=default, 0=no default)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            protected_train: Protected attributes for training set
            protected_val: Protected attributes for validation set

        Returns:
            Dictionary of performance metrics
        """
        logger.info(f"Starting HIGH-RISK model training with {len(X_train)} samples")
        logger.info(f"Default rate in training: {y_train.mean()*100:.2f}%")

        # Store feature names
        self.feature_names = X_train.columns.tolist()

        # Fit scaler on training data
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Create and train model
        self.model = self.create_model()
        self.model.fit(X_train_scaled, y_train)

        # Record training metadata
        self.metadata['trained_at'] = datetime.now().isoformat()
        self.metadata['training_samples'] = len(X_train)
        self.metadata['feature_count'] = len(self.feature_names)
        self.metadata['default_rate_train'] = float(y_train.mean())

        # Evaluate on training set
        train_proba = self.model.predict_proba(X_train_scaled)[:, 1]
        train_predictions = (train_proba >= self.decision_threshold).astype(int)

        # Detailed training metrics
        train_report = classification_report(y_train, train_predictions, output_dict=True)

        metrics = {
            'train_auc': float(roc_auc_score(y_train, train_proba)),
            'train_precision': float(train_report['1']['precision']),
            'train_recall': float(train_report['1']['recall']),
            'train_f1': float(train_report['1']['f1-score']),
            'train_accuracy': float(train_report['accuracy']),
            # Business metrics on training data
            'train_approval_rate': float(1 - train_predictions.mean()),  # % approved
            'train_default_capture_rate': float(train_report['1']['recall'])  # % of defaults caught
        }

        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            val_proba = self.model.predict_proba(X_val_scaled)[:, 1]
            val_predictions = (val_proba >= self.decision_threshold).astype(int)

            metrics['val_auc'] = float(roc_auc_score(y_val, val_proba))

            # Detailed classification metrics
            report = classification_report(y_val, val_predictions, output_dict=True)
            metrics['val_precision'] = float(report['1']['precision'])
            metrics['val_recall'] = float(report['1']['recall'])
            metrics['val_f1'] = float(report['1']['f1-score'])
            metrics['val_accuracy'] = float(report['accuracy'])

            # Business metrics
            metrics['val_approval_rate'] = float(1 - val_predictions.mean())  # % approved
            metrics['val_default_capture_rate'] = float(report['1']['recall'])  # % of defaults caught

            logger.info(f"Validation AUC: {metrics['val_auc']:.4f}")
            logger.info(f"Approval Rate: {metrics['val_approval_rate']*100:.2f}%")

            # Fairness analysis if protected attributes provided
            if protected_val is not None:
                fairness_metrics = self.analyze_fairness(
                    val_predictions, val_proba, y_val, protected_val
                )
                self.metadata['fairness_metrics'] = fairness_metrics
                logger.info("Fairness analysis completed")

        # Store metrics
        self.metadata['performance_metrics'] = metrics

        # Model interpretability: feature coefficients
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.model.coef_[0],
            'abs_coefficient': np.abs(self.model.coef_[0])
        }).sort_values('abs_coefficient', ascending=False)

        self.metadata['feature_coefficients'] = feature_importance.to_dict('records')

        logger.info(f"Top 5 influential features: {feature_importance.head()['feature'].tolist()}")

        return metrics

    def analyze_fairness(
        self,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        y_true: np.ndarray,
        protected_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Analyze model fairness across protected groups.

        Critical for high-risk credit models to ensure Equal Credit Opportunity Act compliance.

        Args:
            predictions: Model predictions
            probabilities: Prediction probabilities
            y_true: True labels
            protected_df: DataFrame with protected attributes

        Returns:
            Dictionary of fairness metrics
        """
        fairness_metrics = {}

        for attr in self.protected_attributes:
            if attr not in protected_df.columns:
                continue

            groups = protected_df[attr].unique()
            attr_metrics = {}

            for group in groups:
                mask = protected_df[attr] == group
                group_preds = predictions[mask]
                group_proba = probabilities[mask]
                group_true = y_true[mask]

                # Approval rate (demographic parity)
                approval_rate = float(1 - group_preds.mean())

                # True positive rate (equal opportunity)
                if group_true.sum() > 0:
                    tpr = float(((group_preds == 1) & (group_true == 1)).sum() / group_true.sum())
                else:
                    tpr = None

                # False positive rate (equalized odds)
                if (group_true == 0).sum() > 0:
                    fpr = float(((group_preds == 1) & (group_true == 0)).sum() / (group_true == 0).sum())
                else:
                    fpr = None

                attr_metrics[str(group)] = {
                    'approval_rate': approval_rate,
                    'true_positive_rate': tpr,
                    'false_positive_rate': fpr,
                    'sample_size': int(mask.sum())
                }

            fairness_metrics[attr] = attr_metrics

            # Calculate disparate impact ratios
            approval_rates = [m['approval_rate'] for m in attr_metrics.values()]
            if len(approval_rates) > 1:
                max_rate = max(approval_rates)
                min_rate = min(approval_rates)
                disparate_impact = min_rate / max_rate if max_rate > 0 else 1.0
                fairness_metrics[f'{attr}_disparate_impact'] = float(disparate_impact)

                # Flag if below 80% threshold (four-fifths rule)
                if disparate_impact < 0.8:
                    logger.warning(
                        f"FAIRNESS ALERT: {attr} disparate impact = {disparate_impact:.3f} (below 0.8 threshold)"
                    )

        return fairness_metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict default risk (binary classification).

        Args:
            X: Feature dataframe

        Returns:
            Array of predictions (0 = approve, 1 = deny)
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        probabilities = self.predict_proba(X)
        predictions = (probabilities >= self.decision_threshold).astype(int)

        logger.info(f"Generated predictions for {len(X)} applicants")
        logger.info(f"Denial rate: {predictions.mean()*100:.2f}%")

        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict default probability.

        Args:
            X: Feature dataframe

        Returns:
            Array of default probabilities
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        X_scaled = self.scaler.transform(X)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]

        return probabilities

    def explain_prediction(self, X: pd.DataFrame, idx: int = 0) -> Dict[str, Any]:
        """
        Explain a specific prediction (required for high-risk models).

        Args:
            X: Feature dataframe
            idx: Index of prediction to explain

        Returns:
            Dictionary with explanation details
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        # Get prediction
        proba = self.predict_proba(X.iloc[[idx]])[0]
        decision = "DENY" if proba >= self.decision_threshold else "APPROVE"

        # Feature contributions
        X_scaled = self.scaler.transform(X.iloc[[idx]])
        contributions = X_scaled[0] * self.model.coef_[0]

        feature_contributions = pd.DataFrame({
            'feature': self.feature_names,
            'value': X.iloc[idx].values,
            'contribution': contributions
        }).sort_values('contribution', key=abs, ascending=False)

        explanation = {
            'decision': decision,
            'default_probability': float(proba),
            'threshold': self.decision_threshold,
            'top_factors': feature_contributions.head(5).to_dict('records'),
            'all_factors': feature_contributions.to_dict('records')
        }

        return explanation

    def save(self, model_path: str, metadata_path: str = None):
        """
        Save the trained model with comprehensive metadata.

        Args:
            model_path: Path to save model artifacts
            metadata_path: Path to save metadata (optional)
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        artifacts = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'metadata': self.metadata,
            'decision_threshold': self.decision_threshold,
            'protected_attributes': self.protected_attributes
        }

        joblib.dump(artifacts, model_path)
        logger.info(f"HIGH-RISK model saved to {model_path}")

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
        self.decision_threshold = artifacts.get('decision_threshold', 0.5)
        self.protected_attributes = artifacts.get('protected_attributes', [])

        logger.info(f"HIGH-RISK model loaded from {model_path}")
        logger.info(f"Model version: {self.metadata.get('model_version')}")
        logger.info(f"Risk tier: {self.metadata.get('risk_tier')}")


def generate_synthetic_credit_data(n_samples: int = 5000, default_rate: float = 0.15) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generate synthetic credit application data for demonstration.

    Args:
        n_samples: Number of samples to generate
        default_rate: Proportion of defaults

    Returns:
        Tuple of (features, labels)
    """
    np.random.seed(42)

    data = {
        'age': np.random.randint(22, 70, n_samples),
        'annual_income': np.random.lognormal(10.5, 0.8, n_samples),
        'years_employed': np.random.randint(0, 40, n_samples),
        'credit_score': np.random.normal(680, 80, n_samples).clip(300, 850),
        'credit_limit': np.random.lognormal(9, 0.8, n_samples),
        'credit_used': np.random.lognormal(8, 1.0, n_samples),
        'monthly_debt': np.random.lognormal(7, 0.8, n_samples),
        'loan_amount': np.random.lognormal(10, 0.6, n_samples),
        'num_credit_inquiries': np.random.poisson(2, n_samples),
        'num_delinquencies': np.random.poisson(0.5, n_samples),
        'gender': np.random.choice(['M', 'F'], n_samples),
    }

    df = pd.DataFrame(data)

    # Generate labels based on features (simulate realistic defaults)
    default_score = (
        -0.01 * df['credit_score'] +
        -0.00001 * df['annual_income'] +
        0.3 * df['num_delinquencies'] +
        0.1 * df['num_credit_inquiries'] +
        np.random.normal(0, 2, n_samples)
    )

    # Convert scores to probabilities
    default_proba = 1 / (1 + np.exp(-default_score))
    labels = (default_proba > (1 - default_rate)).astype(int)

    logger.info(f"Generated {n_samples} synthetic credit applications ({labels.mean()*100:.1f}% default)")

    return df, pd.Series(labels)


if __name__ == "__main__":
    """
    Example usage of the credit risk model with governance best practices.
    """

    print("="*70)
    print("CREDIT RISK MODEL - HIGH RISK TIER")
    print("Demonstrates comprehensive governance for financial models")
    print("="*70)

    # Generate synthetic data
    print("\n[1/7] Generating synthetic credit application data...")
    data, labels = generate_synthetic_credit_data(n_samples=5000, default_rate=0.15)

    # Initialize model
    print("\n[2/7] Initializing HIGH-RISK credit model...")
    model = CreditRiskModel(random_state=42)

    # Validate data quality
    print("\n[3/7] Validating data quality...")
    validation_results = model.validate_data_quality(data)
    if not validation_results['passed']:
        print("  ⚠️  Data quality issues detected:")
        for issue in validation_results['issues']:
            print(f"      - {issue}")

    # Prepare features
    print("\n[4/7] Engineering features...")
    features_df = model.prepare_features(data)

    # Select features for training (exclude protected attributes)
    feature_cols = [
        'annual_income', 'years_employed', 'credit_score',
        'num_credit_inquiries', 'num_delinquencies',
        'debt_to_income', 'credit_utilization',
        'income_stability', 'loan_to_income'
    ]
    protected_cols = ['age_group', 'gender']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features_df[feature_cols], labels, test_size=0.2, random_state=42, stratify=labels
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # Split protected attributes
    _, test_protected = train_test_split(
        features_df[protected_cols], test_size=0.2, random_state=42, stratify=labels
    )
    _, val_protected = train_test_split(
        features_df[protected_cols][:(len(features_df)-len(test_protected))],
        test_size=0.2, random_state=42,
        stratify=labels[:(len(labels)-len(test_protected))]
    )

    # Train model
    print("\n[5/7] Training model with fairness analysis...")
    metrics = model.train(
        X_train, y_train,
        X_val, y_val,
        protected_val=val_protected
    )

    print(f"\n  Performance Metrics:")
    print(f"    Train AUC:     {metrics['train_auc']:.4f}")
    print(f"    Val AUC:       {metrics['val_auc']:.4f}")
    print(f"    Val Precision: {metrics['val_precision']:.4f}")
    print(f"    Val Recall:    {metrics['val_recall']:.4f}")
    print(f"    Val F1:        {metrics['val_f1']:.4f}")
    print(f"    Approval Rate: {metrics['val_approval_rate']*100:.2f}%")

    # Test model
    print("\n[6/7] Evaluating on test set...")
    test_predictions = model.predict(X_test)
    test_proba = model.predict_proba(X_test)

    test_auc = roc_auc_score(y_test, test_proba)
    test_approval_rate = 1 - test_predictions.mean()

    print(f"    Test AUC:       {test_auc:.4f}")
    print(f"    Approval Rate:  {test_approval_rate*100:.2f}%")

    # Confusion matrix
    cm = confusion_matrix(y_test, test_predictions)
    print(f"\n  Confusion Matrix:")
    print(f"    TN: {cm[0,0]:4d}  FP: {cm[0,1]:4d}")
    print(f"    FN: {cm[1,0]:4d}  TP: {cm[1,1]:4d}")

    # Fairness analysis on test set
    print("\n  Fairness Analysis (Test Set):")
    test_fairness = model.analyze_fairness(test_predictions, test_proba, y_test.values, test_protected)

    for attr, metrics_dict in test_fairness.items():
        if isinstance(metrics_dict, dict) and 'approval_rate' in list(metrics_dict.values())[0]:
            print(f"\n    {attr}:")
            for group, group_metrics in metrics_dict.items():
                print(f"      {group}: {group_metrics['approval_rate']*100:.1f}% approval rate (n={group_metrics['sample_size']})")

    # Example prediction explanation
    print("\n[7/7] Example prediction explanation...")
    explanation = model.explain_prediction(X_test, idx=0)
    print(f"    Decision: {explanation['decision']}")
    print(f"    Default Probability: {explanation['default_probability']:.2%}")
    print(f"    Top Contributing Factors:")
    for factor in explanation['top_factors'][:3]:
        print(f"      - {factor['feature']}: {factor['value']:.2f} (contribution: {factor['contribution']:.3f})")

    # Save model
    print("\n[FINAL] Saving model and metadata...")
    model.save('credit_risk_model.pkl', 'credit_risk_metadata.json')

    print("\n" + "="*70)
    print("✅ HIGH-RISK model training complete!")
    print("   Next steps:")
    print("   1. Complete model card using template")
    print("   2. Conduct independent validation")
    print("   3. Obtain required approvals (Validator + Compliance + Business Owner)")
    print("   4. Register in model registry")
    print("   5. Set up production monitoring")
    print("="*70)
