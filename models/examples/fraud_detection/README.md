# Fraud Detection Model Example

This directory contains a complete example of a fraud detection model with proper governance practices.

## Overview

- **Model ID:** fraud-detection-v1
- **Model Type:** Random Forest Classifier
- **Risk Tier:** Medium
- **Purpose:** Detect fraudulent transactions in real-time

## Files

- `fraud_detection_model.py` - Main model implementation
- `README.md` - This file
- `requirements.txt` - Python dependencies

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Training the Model

```bash
python fraud_detection_model.py
```

This will:
1. Generate synthetic transaction data
2. Train the fraud detection model
3. Evaluate performance on validation and test sets
4. Save the trained model to `fraud_detection_model.pkl`

### Using the Model

```python
from fraud_detection_model import FraudDetectionModel
import pandas as pd

# Load trained model
model = FraudDetectionModel()
model.load('fraud_detection_model.pkl')

# Prepare transaction data
transactions = pd.DataFrame({
    'timestamp': [...],
    'user_id': [...],
    'amount': [...],
    # ... other features
})

# Engineer features
features = model.prepare_features(transactions)

# Get predictions
predictions = model.predict(features[feature_cols])
probabilities = model.predict_proba(features[feature_cols])
```

## Model Features

The model uses the following features:

1. **log_amount** - Log-transformed transaction amount
2. **hour_of_day** - Hour when transaction occurred (0-23)
3. **day_of_week** - Day of week (0-6)
4. **is_weekend** - Binary flag for weekend transactions
5. **location_change** - Flag indicating unusual location

## Performance Metrics

Expected performance on synthetic data:
- **AUC-ROC:** ~0.85-0.90
- **Precision:** ~0.70-0.80
- **Recall:** ~0.65-0.75

Note: These metrics are on synthetic data. Real-world performance will vary.

## Governance Compliance

This example demonstrates:

✅ **Reproducibility**
- Fixed random seeds for deterministic results
- Versioned dependencies
- Complete training code

✅ **Logging**
- Comprehensive logging of training and inference
- Audit trail of model decisions

✅ **Documentation**
- Inline code documentation
- Clear model card and technical docs
- Performance metrics tracked

✅ **Monitoring**
- Metadata tracking
- Performance metrics logged
- Ready for production monitoring

✅ **Bias Considerations**
- Class imbalance handled with balanced weights
- Feature importance tracked
- Designed for fairness analysis

## Next Steps

1. **Create Model Card** - Use template in `/docs/templates/model_card.md`
2. **Add to Registry** - Update `/inventory/model_registry.yaml`
3. **Set Up Monitoring** - Implement drift detection and performance tracking
4. **Conduct Validation** - Complete validation report using template

## Integration with Governance Framework

To integrate this model with the governance framework:

1. **Register the model** in `inventory/model_registry.yaml`:
```yaml
- model_id: fraud-detection-v1
  name: "Fraud Detection Model"
  version: "1.0.0"
  risk_tier: medium
  # ... other fields
```

2. **Create documentation** using templates in `docs/templates/`

3. **Run validation** before deployment

4. **Set up CI/CD** pipeline to run automated tests

## License

This is example code for demonstration purposes.
