# Test Suite

This directory contains tests for the AI Governance Starter Kit.

## Test Organization

### `test_registry_validation.py`
Tests for model registry validation:
- Registry structure and format
- Required fields presence
- Risk tier validation
- Owner information completeness
- Deployment configuration
- Monitoring setup
- High-risk model requirements
- Documentation completeness

### `test_model_performance.py`
Tests for model performance and quality:
- Performance metrics (AUC, precision, recall)
- Model stability and reproducibility
- Data validation
- Fairness metrics
- Model artifact saving/loading

## Running Tests

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test File
```bash
pytest tests/test_registry_validation.py -v
```

### Run Specific Test Class
```bash
pytest tests/test_registry_validation.py::TestRiskTier -v
```

### Run Specific Test
```bash
pytest tests/test_registry_validation.py::TestRiskTier::test_valid_risk_tier -v
```

### Run with Coverage
```bash
pytest tests/ --cov=. --cov-report=html
```

## Test Categories

### Registry Validation Tests
These tests ensure your model registry follows governance standards:

✅ **Structure Tests**
- Registry file exists and is valid YAML
- Contains required top-level keys
- Models list is properly formatted

✅ **Required Fields**
- All models have mandatory fields
- Model IDs are unique
- Naming conventions followed

✅ **Risk Tier Tests**
- Risk tiers are valid values
- High-risk models have additional requirements
- Governance policies align with risk tiers

✅ **Documentation Tests**
- Production models have documentation
- High-risk models have validation reports
- Documentation links are present

### Model Performance Tests
These tests ensure models meet quality standards:

✅ **Performance Tests**
- Models perform better than baseline
- Meet minimum performance thresholds
- Precision/recall are balanced
- No extreme overfitting

✅ **Stability Tests**
- Models are reproducible
- Performance stable across data splits
- Results consistent with same seed

✅ **Data Quality Tests**
- No missing values in features
- Feature types are consistent
- Target distribution checked

✅ **Fairness Tests**
- Demographic parity calculated
- Equal opportunity metrics computed
- Protected groups analyzed

## Writing New Tests

### Example: Testing a New Model

```python
import pytest
from models.examples.your_model import YourModel

class TestYourModel:
    """Tests for your custom model."""

    @pytest.fixture
    def model(self):
        """Create model instance for testing."""
        return YourModel(random_state=42)

    def test_model_initialization(self, model):
        """Test that model initializes correctly."""
        assert model is not None
        assert model.random_state == 42

    def test_model_training(self, model):
        """Test that model can be trained."""
        X_train = [[1, 2], [3, 4], [5, 6]]
        y_train = [0, 1, 0]

        model.train(X_train, y_train)

        assert model.model is not None

    def test_model_prediction(self, model):
        """Test that trained model can make predictions."""
        X_train = [[1, 2], [3, 4], [5, 6]]
        y_train = [0, 1, 0]

        model.train(X_train, y_train)

        X_test = [[2, 3]]
        predictions = model.predict(X_test)

        assert predictions is not None
        assert len(predictions) == 1
```

### Example: Testing Registry Entry

```python
def test_my_model_in_registry(models):
    """Test that my model is properly registered."""
    my_model = next(
        (m for m in models if m['model_id'] == 'my-model-v1'),
        None
    )

    assert my_model is not None, "my-model-v1 not found in registry"
    assert my_model['risk_tier'] == 'medium'
    assert 'validator' in my_model['owner']
```

## CI/CD Integration

These tests are designed to run in the CI/CD pipeline:

```yaml
# In .github/workflows/risk-gate.yml
- name: Run Tests
  run: |
    pip install pytest pytest-cov
    pytest tests/ -v --cov=. --cov-report=xml

- name: Upload Coverage
  uses: codecov/codecov-action@v3
  with:
    files: ./coverage.xml
```

## Test Fixtures

Common fixtures are available in `conftest.py`:
- `registry_data` - Loaded registry YAML
- `models` - List of models from registry
- `synthetic_data` - Generated test data

## Best Practices

### 1. Test Independence
Each test should be independent and not rely on others:
```python
# Good
def test_something():
    data = generate_test_data()
    result = process(data)
    assert result == expected

# Bad - relies on global state
global_data = None
def test_step1():
    global global_data
    global_data = generate_test_data()

def test_step2():
    result = process(global_data)  # Depends on test_step1
```

### 2. Use Descriptive Names
```python
# Good
def test_high_risk_model_requires_validation_report():
    ...

# Less clear
def test_validation():
    ...
```

### 3. Test One Thing
```python
# Good - focused test
def test_model_id_is_unique():
    model_ids = [m['model_id'] for m in models]
    assert len(model_ids) == len(set(model_ids))

# Bad - testing multiple things
def test_model_registry():
    # Tests uniqueness, format, required fields all in one
    ...
```

### 4. Use Parametrize for Similar Tests
```python
@pytest.mark.parametrize("field", ['model_id', 'name', 'version'])
def test_required_field(models, field):
    for model in models:
        assert field in model
```

## Continuous Improvement

As you add new models or governance requirements:

1. **Update Registry Tests** - Add tests for new required fields
2. **Add Model-Specific Tests** - Test unique model behaviors
3. **Expand Fairness Tests** - Add tests for new protected attributes
4. **Monitor Test Coverage** - Aim for >80% code coverage

## Troubleshooting

### Tests Failing?

**Registry validation tests failing:**
- Check your `model_registry.yaml` syntax
- Ensure all required fields are present
- Verify risk tiers are lowercase

**Model performance tests failing:**
- Check if model meets minimum thresholds
- Verify random seeds are set
- Ensure data quality is good

**Import errors:**
- Install test dependencies: `pip install -r requirements-dev.txt`
- Check Python path includes project root

### Getting Help

If tests are failing and you're not sure why:
1. Run with verbose output: `pytest -vv`
2. Run single test to isolate issue: `pytest tests/test_file.py::test_name`
3. Check test output for specific error messages
4. Review the test code to understand what's being checked

## License

Tests are part of the AI Governance Starter Kit.
