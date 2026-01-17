"""
Tests for model registry validation.

These tests ensure the model registry YAML file contains valid entries
with all required fields and proper structure.
"""

import pytest
import yaml
from pathlib import Path
from typing import Dict, Any, List


# Path to registry file
REGISTRY_PATH = Path(__file__).parent.parent / "inventory" / "model_registry.yaml"


@pytest.fixture
def registry_data() -> Dict[str, Any]:
    """Load the model registry YAML file."""
    with open(REGISTRY_PATH, 'r') as f:
        data = yaml.safe_load(f)
    return data


@pytest.fixture
def models(registry_data) -> List[Dict]:
    """Extract the models list from registry."""
    return registry_data.get('models', [])


class TestRegistryStructure:
    """Test the overall structure of the registry."""

    def test_registry_file_exists(self):
        """Test that the registry file exists."""
        assert REGISTRY_PATH.exists(), f"Registry file not found at {REGISTRY_PATH}"

    def test_registry_is_valid_yaml(self):
        """Test that the registry is valid YAML."""
        with open(REGISTRY_PATH, 'r') as f:
            try:
                yaml.safe_load(f)
            except yaml.YAMLError as e:
                pytest.fail(f"Invalid YAML: {e}")

    def test_registry_has_models_key(self, registry_data):
        """Test that registry contains 'models' key."""
        assert 'models' in registry_data, "Registry must contain 'models' key"

    def test_models_is_list(self, registry_data):
        """Test that models is a list."""
        assert isinstance(registry_data['models'], list), "'models' must be a list"

    def test_registry_not_empty(self, models):
        """Test that registry contains at least one model."""
        assert len(models) > 0, "Registry must contain at least one model"


class TestRequiredFields:
    """Test that all models have required fields."""

    REQUIRED_FIELDS = [
        'model_id',
        'name',
        'version',
        'risk_tier',
        'owner',
        'description',
        'model_type',
        'framework',
        'deployment',
        'monitoring',
    ]

    @pytest.mark.parametrize("field", REQUIRED_FIELDS)
    def test_model_has_required_field(self, models, field):
        """Test that all models have required top-level fields."""
        for i, model in enumerate(models):
            assert field in model, f"Model {i} ({model.get('model_id', 'unknown')}) missing required field: {field}"

    def test_model_id_is_unique(self, models):
        """Test that all model_ids are unique."""
        model_ids = [m['model_id'] for m in models]
        duplicates = [mid for mid in model_ids if model_ids.count(mid) > 1]
        assert len(duplicates) == 0, f"Duplicate model_ids found: {set(duplicates)}"

    def test_model_id_format(self, models):
        """Test that model_ids follow naming convention."""
        for model in models:
            model_id = model['model_id']
            # Should be lowercase, hyphen-separated, end with version
            assert model_id.islower() or '-' in model_id, \
                f"model_id should be lowercase: {model_id}"


class TestRiskTier:
    """Test risk tier classification and requirements."""

    VALID_RISK_TIERS = ['low', 'medium', 'high', 'critical']

    def test_valid_risk_tier(self, models):
        """Test that risk_tier is one of the valid values."""
        for model in models:
            risk_tier = model['risk_tier']
            assert risk_tier in self.VALID_RISK_TIERS, \
                f"Invalid risk_tier '{risk_tier}' in model {model['model_id']}. " \
                f"Must be one of: {self.VALID_RISK_TIERS}"

    def test_risk_tier_is_lowercase(self, models):
        """Test that risk_tier values are lowercase."""
        for model in models:
            risk_tier = model['risk_tier']
            assert risk_tier.islower(), \
                f"risk_tier must be lowercase: {risk_tier} in {model['model_id']}"


class TestOwnerInformation:
    """Test owner information completeness."""

    REQUIRED_OWNER_FIELDS = ['team', 'technical_lead', 'validator']

    def test_owner_has_required_fields(self, models):
        """Test that owner section has all required fields."""
        for model in models:
            owner = model.get('owner', {})
            for field in self.REQUIRED_OWNER_FIELDS:
                assert field in owner, \
                    f"Model {model['model_id']} owner missing required field: {field}"

    def test_email_format(self, models):
        """Test that email addresses in owner contain @."""
        for model in models:
            owner = model.get('owner', {})
            if 'technical_lead' in owner and owner['technical_lead']:
                email = owner['technical_lead']
                if isinstance(email, str):
                    assert '@' in email, \
                        f"Invalid email in {model['model_id']}: {email}"


class TestDeployment:
    """Test deployment configuration."""

    VALID_ENVIRONMENTS = ['development', 'staging', 'production', 'test']

    def test_deployment_has_environment(self, models):
        """Test that deployment section has environment."""
        for model in models:
            deployment = model.get('deployment', {})
            assert 'environment' in deployment, \
                f"Model {model['model_id']} deployment missing 'environment'"

    def test_valid_environment(self, models):
        """Test that environment is valid."""
        for model in models:
            deployment = model.get('deployment', {})
            if 'environment' in deployment:
                env = deployment['environment']
                assert env in self.VALID_ENVIRONMENTS, \
                    f"Invalid environment '{env}' in {model['model_id']}. " \
                    f"Must be one of: {self.VALID_ENVIRONMENTS}"


class TestMonitoring:
    """Test monitoring configuration."""

    def test_monitoring_has_metrics(self, models):
        """Test that monitoring section defines metrics."""
        for model in models:
            monitoring = model.get('monitoring', {})
            # Should have at least one of: performance_metrics, drift_detection, alerts
            has_monitoring = any(key in monitoring for key in [
                'performance_metrics',
                'drift_detection',
                'alerts',
                'review_schedule'
            ])
            assert has_monitoring, \
                f"Model {model['model_id']} monitoring configuration is empty or missing"


class TestHighRiskRequirements:
    """Test that high-risk models have additional required fields."""

    def test_high_risk_has_validation_report(self, models):
        """Test that high/critical risk models have validation reports."""
        for model in models:
            if model['risk_tier'] in ['high', 'critical']:
                docs = model.get('documentation', {})
                # Should have validation report or note it's pending
                has_validation = 'validation_report' in docs
                assert has_validation or model.get('status') == 'development', \
                    f"High/Critical risk model {model['model_id']} missing validation_report"

    def test_high_risk_has_bias_testing(self, models):
        """Test that high/critical risk models have bias testing documented."""
        for model in models:
            if model['risk_tier'] in ['high', 'critical']:
                testing = model.get('testing', {})
                # Should mention bias testing
                has_bias_testing = 'bias_testing' in testing or 'fairness_metrics' in testing
                assert has_bias_testing or model.get('status') == 'development', \
                    f"High/Critical risk model {model['model_id']} missing bias testing info"


class TestVersioning:
    """Test version format and consistency."""

    def test_version_format(self, models):
        """Test that version follows semantic versioning (X.Y.Z)."""
        import re
        version_pattern = r'^\d+\.\d+\.\d+$'

        for model in models:
            version = model.get('version', '')
            assert re.match(version_pattern, version), \
                f"Invalid version format '{version}' in {model['model_id']}. " \
                f"Expected semantic versioning (e.g., 1.0.0)"


class TestDocumentation:
    """Test documentation completeness."""

    def test_has_documentation_section(self, models):
        """Test that models have documentation section."""
        for model in models:
            assert 'documentation' in model, \
                f"Model {model['model_id']} missing 'documentation' section"

    def test_production_models_have_docs(self, models):
        """Test that production models have complete documentation."""
        for model in models:
            if model.get('deployment', {}).get('environment') == 'production':
                docs = model.get('documentation', {})
                # Production models should have model card at minimum
                assert 'model_card' in docs, \
                    f"Production model {model['model_id']} missing model_card"


def test_registry_governance_metadata(registry_data):
    """Test that registry has governance metadata."""
    assert 'governance_policies' in registry_data, \
        "Registry should define governance_policies"

    policies = registry_data.get('governance_policies', {})
    assert 'approval_workflows' in policies, \
        "Governance policies should define approval_workflows"


def test_risk_tier_governance_alignment(registry_data):
    """Test that governance policies align with risk tiers."""
    policies = registry_data.get('governance_policies', {})
    approval_workflows = policies.get('approval_workflows', {})

    # Each risk tier should have defined approval requirements
    for tier in ['low', 'medium', 'high', 'critical']:
        assert tier in approval_workflows, \
            f"Approval workflow not defined for risk tier: {tier}"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, '-v'])
