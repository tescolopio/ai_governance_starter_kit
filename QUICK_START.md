# Quick Start Guide

Get up and running with the AI Governance Starter Kit in 5 minutes.

## Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Git

## Installation

### Option 1: Automated Setup (Recommended)

```bash
# Clone the repository (if not already done)
git clone <repository-url>
cd ai_governance_starter_kit

# Run the setup script
bash scripts/setup.sh
```

The setup script will:
1. Check your Python version
2. Install all dependencies
3. Set up pre-commit hooks
4. Validate the model registry
5. Run tests to verify installation

### Option 2: Manual Setup

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install

# Validate everything is working
make validate
make test
```

## Quick Tour

### 1. Explore Example Models

We've included two fully-implemented example models:

**Fraud Detection (Medium Risk)**
```bash
cd models/examples/fraud_detection
python fraud_detection_model.py
```

**Credit Risk (High Risk)**
```bash
cd models/examples/credit_risk
python credit_risk_model.py
```

### 2. Understand the Registry

View all registered models:
```bash
make list-models
```

Validate the registry:
```bash
make validate
```

The registry is in `inventory/model_registry.yaml` - this is your central source of truth for all AI/ML models.

### 3. Run Tests

Run all tests:
```bash
make test
```

Run specific test suites:
```bash
make test-registry   # Registry validation tests
make test-models     # Model performance tests
```

### 4. Use Documentation Templates

Ready-to-use templates are in `docs/templates/`:

- **model_card.md** - Document your model's purpose, performance, and limitations
- **technical_documentation.md** - Detailed technical specifications
- **validation_report.md** - Independent validation results
- **risk_assessment_worksheet.md** - Assess and classify model risk

To create documentation for your model:
```bash
cp docs/templates/model_card.md docs/my-model-card.md
# Edit the file with your model's information
```

### 5. Add Your First Model

#### Step 1: Assess Risk Tier
Use the risk assessment worksheet:
```bash
# Fill out docs/templates/risk_assessment_worksheet.md
```

#### Step 2: Register the Model
Add an entry to `inventory/model_registry.yaml`:

```yaml
- model_id: my-model-v1
  name: "My ML Model"
  version: "1.0.0"
  risk_tier: medium  # Based on your risk assessment
  owner:
    team: "Data Science"
    technical_lead: "your.email@company.com"
    validator: "validator@company.com"
  # ... other required fields
```

#### Step 3: Create Documentation
Based on your risk tier, create required docs:
- **Low:** model_card
- **Medium:** model_card + technical_docs
- **High/Critical:** model_card + technical_docs + validation_report

#### Step 4: Validate
```bash
make validate
```

## Common Commands

### Validation & Testing
```bash
make validate        # Validate model registry
make test            # Run all tests
make test-coverage   # Run tests with coverage report
make check-docs      # Check documentation completeness
```

### Code Quality
```bash
make lint            # Run linting checks
make format          # Format code
make pre-commit      # Run all pre-commit hooks
```

### Model Operations
```bash
make train-examples  # Train example models
make list-models     # List registered models
```

### Cleanup
```bash
make clean           # Remove cache files
make clean-all       # Deep clean (including model artifacts)
```

### Help
```bash
make help            # Show all available commands
```

## Understanding Risk Tiers

The starter kit uses a 4-tier risk classification:

### Low Risk
- Minimal business impact
- Informational/analytics only
- **Requirements:** Basic model card
- **Validation:** Annual review

### Medium Risk
- Moderate business impact
- Decision support systems
- **Requirements:** Model card + technical docs
- **Validation:** Semi-annual review

### High Risk
- Significant business impact
- Automated decisions affecting customers
- Regulatory requirements
- **Requirements:** Full documentation suite
- **Validation:** Quarterly review
- **Approvals:** Validator + Compliance + Business Owner

### Critical Risk
- Mission-critical systems
- Severe consequences if wrong
- Heavy regulatory oversight
- **Requirements:** Enhanced documentation
- **Validation:** Monthly review
- **Approvals:** CRO + Compliance + Validation team

## Workflows

### Deploying a Low-Risk Model

1. Train model
2. Create model card
3. Register in `model_registry.yaml`
4. Get technical lead approval
5. Deploy

### Deploying a High-Risk Model

1. Train model with proper governance (see examples)
2. Create full documentation suite
3. Conduct independent validation
4. Complete validation report
5. Register in `model_registry.yaml`
6. Get multi-level approvals
7. Set up production monitoring
8. Deploy through CI/CD pipeline

## Project Structure

```
ai_governance_starter_kit/
├── README.md                    # Main documentation
├── QUICK_START.md              # This file
├── Makefile                    # Common commands
├── requirements.txt            # Runtime dependencies
├── requirements-dev.txt        # Development dependencies
├── .pre-commit-config.yaml     # Pre-commit hooks
│
├── inventory/
│   └── model_registry.yaml     # Central model registry
│
├── docs/
│   └── templates/              # Documentation templates
│       ├── model_card.md
│       ├── technical_documentation.md
│       ├── validation_report.md
│       └── risk_assessment_worksheet.md
│
├── models/
│   └── examples/               # Example model implementations
│       ├── fraud_detection/    # Medium-risk example
│       └── credit_risk/        # High-risk example
│
├── tests/                      # Test suite
│   ├── test_registry_validation.py
│   ├── test_model_performance.py
│   └── README.md
│
├── scripts/                    # Utility scripts
│   ├── setup.sh
│   ├── validate_registry.py
│   └── check_documentation.py
│
├── validation/
│   └── structural/
│       └── code_scalpel_config.yaml  # Code auditing rules
│
└── .github/
    └── workflows/
        └── risk-gate.yml       # CI/CD pipeline
```

## Next Steps

### For Small Teams

1. **Start Small**
   - Register existing models in the registry
   - Create basic model cards
   - Set up the CI/CD pipeline

2. **Build Gradually**
   - Add documentation for high-risk models
   - Implement monitoring for critical models
   - Expand test coverage

3. **Mature Over Time**
   - Integrate with your existing tools (Slack, JIRA, etc.)
   - Customize risk tiers for your organization
   - Add custom validation rules

### For Individual Professionals

1. **Learn the Framework**
   - Train the example models
   - Review the documentation templates
   - Understand the risk classification

2. **Apply to Your Projects**
   - Use templates for your models
   - Adopt the registry pattern
   - Implement pre-commit hooks

3. **Demonstrate Governance**
   - Show model cards in presentations
   - Reference validation reports
   - Use for portfolio/resume

## Getting Help

### Documentation
- Main README: Comprehensive overview
- Test README: `tests/README.md`
- Model Examples: `models/examples/*/README.md`

### Commands
```bash
make help            # Show all commands
pytest tests/ -v     # Run tests with verbose output
python scripts/validate_registry.py --help  # Script help
```

### Common Issues

**Issue: Tests failing after installation**
```bash
# Solution: Make sure you installed dev dependencies
pip install -r requirements-dev.txt
```

**Issue: Pre-commit hooks failing**
```bash
# Solution: Run manually to see errors
pre-commit run --all-files
```

**Issue: Registry validation errors**
```bash
# Solution: Run validator to see specific errors
python scripts/validate_registry.py
```

## Best Practices

✅ **DO:**
- Keep your registry up to date
- Document all production models
- Run tests before committing
- Use risk tiers appropriately
- Review models regularly based on risk tier

❌ **DON'T:**
- Skip documentation for high-risk models
- Deploy without validation
- Ignore test failures
- Bypass governance for "just this once"
- Forget to monitor production models

## Resources

- [SR 11-7 Guidance](https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm)
- [Model Cards Paper](https://arxiv.org/abs/1810.03993)
- [AI Fairness Resources](https://fairlearn.org/)

---

**Ready to go?** Start by running `make help` to see all available commands!
