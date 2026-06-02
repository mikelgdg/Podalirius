# Contributing to Triage-MRI

## Development Environment Setup

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd triage-mri
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```

3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Running Tests

```bash
pytest tests/
```

## Pull Request Process

1. Create a feature branch from `dev`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes, ensuring all tests pass and linting is clean.

3. Write or update tests for any new functionality.

4. Update documentation if your changes affect usage.

5. Submit a pull request to the `main` branch with a clear description of changes.

6. Ensure CI checks pass before requesting review.

## Coding Conventions

- Follow PEP 8 style guide.
- Use type hints for all function signatures.
- Docstrings follow Google style.
- Keep functions small and focused on a single responsibility.
- Use meaningful variable and function names.
- Run `ruff check` and `ruff format` before committing.
