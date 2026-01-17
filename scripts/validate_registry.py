#!/usr/bin/env python3
"""
Validate Model Registry

This script validates the model registry YAML file to ensure it meets
governance requirements. It's used as a pre-commit hook and in CI/CD.
"""

import sys
import yaml
from pathlib import Path
from typing import Dict, List, Any


# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_success(msg: str):
    print(f"{Colors.GREEN}✓{Colors.END} {msg}")


def print_error(msg: str):
    print(f"{Colors.RED}✗{Colors.END} {msg}")


def print_warning(msg: str):
    print(f"{Colors.YELLOW}⚠{Colors.END} {msg}")


def print_info(msg: str):
    print(f"{Colors.BLUE}ℹ{Colors.END} {msg}")


class RegistryValidator:
    """Validates model registry structure and content."""

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

    VALID_RISK_TIERS = ['low', 'medium', 'high', 'critical']

    def __init__(self, registry_path: Path):
        self.registry_path = registry_path
        self.errors = []
        self.warnings = []
        self.registry_data = None

    def validate(self) -> bool:
        """
        Run all validation checks.

        Returns:
            True if validation passes, False otherwise
        """
        print(f"\n{Colors.BOLD}Validating Model Registry{Colors.END}")
        print(f"File: {self.registry_path}\n")

        # Load and validate YAML
        if not self._load_yaml():
            return False

        # Run validation checks
        self._validate_structure()
        self._validate_models()
        self._validate_governance_policies()

        # Print results
        self._print_results()

        return len(self.errors) == 0

    def _load_yaml(self) -> bool:
        """Load and parse YAML file."""
        if not self.registry_path.exists():
            self.errors.append(f"Registry file not found: {self.registry_path}")
            return False

        try:
            with open(self.registry_path, 'r') as f:
                self.registry_data = yaml.safe_load(f)
            print_success("YAML file loaded successfully")
            return True
        except yaml.YAMLError as e:
            self.errors.append(f"Invalid YAML syntax: {e}")
            return False

    def _validate_structure(self):
        """Validate top-level registry structure."""
        if 'models' not in self.registry_data:
            self.errors.append("Registry must contain 'models' key")
            return

        if not isinstance(self.registry_data['models'], list):
            self.errors.append("'models' must be a list")
            return

        if len(self.registry_data['models']) == 0:
            self.warnings.append("Registry contains no models")
        else:
            print_success(f"Found {len(self.registry_data['models'])} models")

    def _validate_models(self):
        """Validate individual model entries."""
        if 'models' not in self.registry_data:
            return

        models = self.registry_data['models']
        model_ids = []

        for i, model in enumerate(models):
            model_id = model.get('model_id', f'model_{i}')

            # Check for required fields
            for field in self.REQUIRED_FIELDS:
                if field not in model:
                    self.errors.append(
                        f"Model '{model_id}': Missing required field '{field}'"
                    )

            # Check model_id uniqueness
            if 'model_id' in model:
                if model['model_id'] in model_ids:
                    self.errors.append(
                        f"Duplicate model_id: '{model['model_id']}'"
                    )
                model_ids.append(model['model_id'])

            # Validate risk tier
            if 'risk_tier' in model:
                risk_tier = model['risk_tier']
                if risk_tier not in self.VALID_RISK_TIERS:
                    self.errors.append(
                        f"Model '{model_id}': Invalid risk_tier '{risk_tier}'. "
                        f"Must be one of: {self.VALID_RISK_TIERS}"
                    )

                # Check high-risk requirements
                if risk_tier in ['high', 'critical']:
                    self._validate_high_risk_model(model, model_id)

            # Validate owner section
            if 'owner' in model:
                self._validate_owner(model['owner'], model_id)

            # Validate version format
            if 'version' in model:
                self._validate_version(model['version'], model_id)

    def _validate_high_risk_model(self, model: Dict, model_id: str):
        """Validate additional requirements for high-risk models."""
        # Check for validation report (unless in development)
        if model.get('status') != 'development':
            docs = model.get('documentation', {})
            if 'validation_report' not in docs:
                self.warnings.append(
                    f"Model '{model_id}': High/Critical risk model should have validation_report"
                )

            # Check for bias testing
            testing = model.get('testing', {})
            if 'bias_testing' not in testing and 'fairness_metrics' not in testing:
                self.warnings.append(
                    f"Model '{model_id}': High/Critical risk model should document bias testing"
                )

    def _validate_owner(self, owner: Dict, model_id: str):
        """Validate owner information."""
        required_owner_fields = ['team', 'technical_lead', 'validator']

        for field in required_owner_fields:
            if field not in owner:
                self.errors.append(
                    f"Model '{model_id}': Owner missing required field '{field}'"
                )

        # Validate email format (simple check)
        if 'technical_lead' in owner and owner['technical_lead']:
            email = owner['technical_lead']
            if isinstance(email, str) and '@' not in email:
                self.warnings.append(
                    f"Model '{model_id}': technical_lead doesn't appear to be an email: {email}"
                )

    def _validate_version(self, version: str, model_id: str):
        """Validate version format (semantic versioning)."""
        import re
        pattern = r'^\d+\.\d+\.\d+$'

        if not re.match(pattern, version):
            self.errors.append(
                f"Model '{model_id}': Version '{version}' doesn't follow semantic versioning (X.Y.Z)"
            )

    def _validate_governance_policies(self):
        """Validate governance policies section."""
        if 'governance_policies' not in self.registry_data:
            self.warnings.append("Registry missing 'governance_policies' section")
            return

        policies = self.registry_data['governance_policies']

        if 'approval_workflows' not in policies:
            self.warnings.append("Governance policies missing 'approval_workflows'")
        else:
            # Check that all risk tiers have approval workflows
            workflows = policies['approval_workflows']
            for tier in self.VALID_RISK_TIERS:
                if tier not in workflows:
                    self.warnings.append(
                        f"Approval workflow not defined for risk tier: {tier}"
                    )

    def _print_results(self):
        """Print validation results."""
        print("\n" + "=" * 60)

        if len(self.errors) == 0 and len(self.warnings) == 0:
            print_success("All validation checks passed!")
        else:
            if len(self.errors) > 0:
                print(f"\n{Colors.RED}{Colors.BOLD}ERRORS ({len(self.errors)}):{Colors.END}")
                for error in self.errors:
                    print_error(error)

            if len(self.warnings) > 0:
                print(f"\n{Colors.YELLOW}{Colors.BOLD}WARNINGS ({len(self.warnings)}):{Colors.END}")
                for warning in self.warnings:
                    print_warning(warning)

        print("=" * 60 + "\n")


def main():
    """Main entry point."""
    # Find registry file
    registry_path = Path(__file__).parent.parent / "inventory" / "model_registry.yaml"

    # Allow override from command line
    if len(sys.argv) > 1:
        registry_path = Path(sys.argv[1])

    # Validate
    validator = RegistryValidator(registry_path)
    success = validator.validate()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
