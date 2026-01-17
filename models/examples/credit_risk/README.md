# Credit Risk Assessment Model Example

This directory contains a complete example of a HIGH-RISK credit risk assessment model with comprehensive governance practices.

## Overview

- **Model ID:** credit-risk-v2
- **Model Type:** Logistic Regression
- **Risk Tier:** HIGH
- **Purpose:** Assess credit default risk for loan applications
- **Regulatory:** SR 11-7 compliant, Equal Credit Opportunity Act (ECOA)

## Why HIGH RISK?

This model is classified as HIGH RISK because:
- ✅ Makes automated credit decisions affecting customers
- ✅ Financial impact >$1M
- ✅ Subject to regulatory oversight (SR 11-7, ECOA)
- ✅ Uses protected attributes (requires fairness analysis)
- ✅ Affects >10,000 customers

## Files

- `credit_risk_model.py` - Main model implementation with governance features
- `README.md` - This file
- `requirements.txt` - Python dependencies

## Key Governance Features

This example demonstrates comprehensive governance for high-risk models:

### 1. **Interpretability**
- Uses Logistic Regression for full interpretability
- Provides prediction explanations via `explain_prediction()`
- Tracks feature coefficients and importance

### 2. **Fairness & Bias**
- Monitors protected attributes (age, gender)
- Calculates demographic parity, equal opportunity metrics
- Implements four-fifths rule checking
- Flags disparate impact warnings

### 3. **Data Quality**
- Validates data quality before training
- Checks for missing values, duplicates, outliers
- Logs data quality issues

### 4. **Audit Trail**
- Comprehensive logging of all operations
- Metadata tracking (training date, samples, metrics)
- Approval workflow tracking

### 5. **Regulatory Compliance**
- SR 11-7 metadata fields
- ECOA fairness requirements
- Approval requirements documented

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Training the Model

```bash
python credit_risk_model.py
```

This will:
1. Generate synthetic credit application data
2. Validate data quality
3. Train the model with fairness analysis
4. Evaluate performance and fairness metrics
5. Demonstrate prediction explanations
6. Save the trained model

### Using the Model

```python
from credit_risk_model import CreditRiskModel
import pandas as pd

# Load trained model
model = CreditRiskModel()
model.load('credit_risk_model.pkl')

# Prepare applicant data
applicants = pd.DataFrame({
    'annual_income': [75000],
    'years_employed': [5],
    'credit_score': [720],
    # ... other features
})

# Get decision
decision = model.predict(applicants)  # 0 = approve, 1 = deny
probability = model.predict_proba(applicants)

# Get explanation
explanation = model.explain_prediction(applicants, idx=0)
print(f"Decision: {explanation['decision']}")
print(f"Default Risk: {explanation['default_probability']:.2%}")
print(f"Top Factors: {explanation['top_factors']}")
```

## Model Features

The model uses these features (protected attributes excluded from training):

**Financial Features:**
1. **annual_income** - Applicant's annual income
2. **years_employed** - Years in current employment
3. **credit_score** - Credit bureau score (300-850)
4. **num_credit_inquiries** - Recent credit inquiries
5. **num_delinquencies** - Historical delinquencies

**Engineered Features:**
6. **debt_to_income** - Monthly debt / monthly income ratio
7. **credit_utilization** - Credit used / credit limit ratio
8. **income_stability** - Employment duration indicator
9. **loan_to_income** - Requested loan / annual income ratio

**Protected Attributes (monitored, not used in training):**
- **age_group** - Age cohort (18-25, 26-35, 36-50, 50+)
- **gender** - Applicant gender

## Performance Metrics

Expected performance on synthetic data:
- **AUC-ROC:** ~0.80-0.85
- **Precision:** ~0.70-0.75
- **Recall:** ~0.65-0.70
- **Approval Rate:** ~75-80%

## Fairness Requirements

For HIGH-RISK credit models, fairness analysis is mandatory:

### Demographic Parity
- Approval rates should be similar across protected groups
- Four-fifths rule: Ratio of approval rates should be ≥0.8

### Equal Opportunity
- True positive rates should be similar across groups
- Qualified applicants approved at similar rates

### Equalized Odds
- Both TPR and FPR should be similar across groups
- Errors distributed fairly

## Governance Workflow

For HIGH-RISK models, follow this workflow:

### 1. Development Phase
- [x] Implement model with interpretability
- [x] Add comprehensive logging
- [x] Include fairness analysis
- [x] Validate data quality
- [x] Document all code

### 2. Validation Phase
- [ ] Create model card (use template)
- [ ] Write technical documentation
- [ ] Conduct independent validation
- [ ] Complete validation report
- [ ] Address any findings

### 3. Approval Phase
Required approvals for HIGH-RISK:
- [ ] Technical Lead/Validator
- [ ] Compliance Officer
- [ ] Business Owner

### 4. Deployment Phase
- [ ] Register in model registry
- [ ] Set up production monitoring
- [ ] Configure drift detection
- [ ] Establish alert thresholds
- [ ] Document incident response plan

### 5. Monitoring Phase
- [ ] Quarterly validation reviews
- [ ] Continuous performance monitoring
- [ ] Ongoing fairness analysis
- [ ] Regular compliance audits

## Required Documentation

For HIGH-RISK models, complete documentation is required:

1. **Model Card** (`docs/templates/model_card.md`)
   - Model overview and purpose
   - Performance metrics
   - Fairness analysis
   - Limitations and risks

2. **Technical Documentation** (`docs/templates/technical_documentation.md`)
   - Architecture details
   - Data pipeline
   - Deployment guide
   - API specification

3. **Validation Report** (`docs/templates/validation_report.md`)
   - Independent validation results
   - Performance assessment
   - Fairness validation
   - Approval signatures

4. **Risk Assessment** (`docs/templates/risk_assessment_worksheet.md`)
   - Risk tier justification
   - Impact analysis
   - Mitigation measures

## Monitoring Requirements

HIGH-RISK models require enhanced monitoring:

### Performance Monitoring
- Track AUC, precision, recall daily
- Alert if AUC drops below 0.75
- Monitor approval rate trends

### Fairness Monitoring
- Calculate fairness metrics weekly
- Alert if disparate impact <0.8
- Track by protected attribute groups

### Data Drift Monitoring
- Monitor input feature distributions
- Alert on significant drift (PSI >0.25)
- Compare to training distribution

### Business Metrics
- Track default rates by cohort
- Monitor false positive/negative costs
- Analyze customer impact

## Regulatory Compliance

### SR 11-7 Requirements
- [x] Model inventory entry
- [x] Development documentation
- [x] Independent validation
- [x] Ongoing monitoring plan
- [x] Quarterly review schedule
- [x] Appropriate governance

### ECOA Requirements
- [x] Fairness analysis conducted
- [x] Protected attributes tracked
- [x] Adverse action reasons available
- [x] Disparate impact testing
- [x] Regular fairness audits

## Integration with Governance Framework

1. **Add to Model Registry** (`inventory/model_registry.yaml`):
```yaml
- model_id: credit-risk-v2
  name: "Credit Risk Assessment Model v2"
  version: "2.0.0"
  risk_tier: high
  owner:
    team: "Credit Risk Analytics"
    technical_lead: "your-name@company.com"
    validator: "validator@company.com"
  # ... complete all HIGH-RISK fields
```

2. **CI/CD Pipeline** will enforce:
   - All documentation complete
   - Validation report approved
   - Three approvers signed off
   - Tests passing

3. **Monitoring** integration:
   - Send metrics to monitoring dashboard
   - Configure alerts to Slack/email
   - Log all predictions for audit trail

## Example Scenarios

### Scenario 1: Approving a Strong Applicant
```
Input: High income, good credit score, low debt
Output: APPROVE (default risk: 5%)
Top Factors:
  - High credit score (+)
  - Low debt-to-income (+)
  - Low credit utilization (+)
```

### Scenario 2: Denying a Risky Applicant
```
Input: Low income, poor credit, high debt
Output: DENY (default risk: 78%)
Top Factors:
  - Multiple delinquencies (-)
  - High credit utilization (-)
  - Low credit score (-)
```

### Scenario 3: Borderline Decision
```
Input: Average credit, moderate debt
Output: APPROVE (default risk: 52%)
Recommendation: Consider manual review
```

## Troubleshooting

### Issue: Fairness metrics show disparate impact
**Solution:**
1. Review feature engineering for proxy variables
2. Adjust decision threshold by group if appropriate
3. Collect more balanced training data
4. Consider fairness-aware training algorithms

### Issue: Model performance degrading
**Solution:**
1. Check for data drift
2. Review recent defaults for pattern changes
3. Consider retraining with recent data
4. Validate data quality

### Issue: Cannot explain predictions
**Solution:**
- Use `explain_prediction()` method
- Review feature coefficients in metadata
- Check which features contributed most

## License

This is example code for demonstration purposes.

## References

- [SR 11-7: Supervisory Guidance on Model Risk Management](https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm)
- [Equal Credit Opportunity Act](https://www.consumerfinance.gov/rules-policy/regulations/1002/)
- [Fairlearn: Fairness in ML](https://fairlearn.org/)
