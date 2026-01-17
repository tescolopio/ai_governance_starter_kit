#!/usr/bin/env python3
"""
Check Model Documentation

This script checks that models referenced in the registry have the required
documentation files. It's used as a pre-commit hook and in CI/CD.
"""

import sys
import yaml
from pathlib import Path
from typing import Dict, List


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


def check_documentation(registry_path: Path) -> bool:
    """
    Check that models have required documentation.

    Args:
        registry_path: Path to model registry YAML

    Returns:
        True if all checks pass, False otherwise
    """
    print(f"\n{Colors.BOLD}Checking Model Documentation{Colors.END}\n")

    # Load registry
    try:
        with open(registry_path, 'r') as f:
            registry = yaml.safe_load(f)
    except Exception as e:
        print_error(f"Failed to load registry: {e}")
        return False

    models = registry.get('models', [])
    errors = []
    warnings = []
    docs_base = registry_path.parent.parent / "docs"

    # Documentation requirements by risk tier
    REQUIRED_DOCS = {
        'critical': ['model_card', 'technical_docs', 'validation_report'],
        'high': ['model_card', 'technical_docs', 'validation_report'],
        'medium': ['model_card', 'technical_docs'],
        'low': ['model_card'],
    }

    for model in models:
        model_id = model.get('model_id', 'unknown')
        risk_tier = model.get('risk_tier', 'low')
        status = model.get('status', 'production')
        docs = model.get('documentation', {})

        # Skip development models
        if status == 'development':
            print_warning(f"{model_id}: Skipping (development status)")
            continue

        # Check required docs for risk tier
        required = REQUIRED_DOCS.get(risk_tier, ['model_card'])

        for doc_type in required:
            if doc_type not in docs:
                errors.append(
                    f"{model_id}: Missing required '{doc_type}' in documentation section"
                )
            else:
                doc_path = docs[doc_type]
                # Check if file exists (if it's a local path)
                if isinstance(doc_path, str) and doc_path.startswith('docs/'):
                    full_path = docs_base.parent / doc_path
                    if not full_path.exists():
                        warnings.append(
                            f"{model_id}: Documentation file not found: {doc_path}"
                        )
                    else:
                        print_success(f"{model_id}: {doc_type} found")

        # Production models should have API docs
        if model.get('deployment', {}).get('environment') == 'production':
            if 'api_docs' not in docs:
                warnings.append(
                    f"{model_id}: Production model should have API documentation"
                )

    # Print results
    print("\n" + "=" * 60)

    if len(errors) == 0 and len(warnings) == 0:
        print_success("All documentation checks passed!")
    else:
        if len(errors) > 0:
            print(f"\n{Colors.RED}{Colors.BOLD}ERRORS ({len(errors)}):{Colors.END}")
            for error in errors:
                print_error(error)

        if len(warnings) > 0:
            print(f"\n{Colors.YELLOW}{Colors.BOLD}WARNINGS ({len(warnings)}):{Colors.END}")
            for warning in warnings:
                print_warning(warning)

    print("=" * 60 + "\n")

    return len(errors) == 0


def main():
    """Main entry point."""
    registry_path = Path(__file__).parent.parent / "inventory" / "model_registry.yaml"

    if len(sys.argv) > 1:
        registry_path = Path(sys.argv[1])

    success = check_documentation(registry_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
