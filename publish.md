## Publishing to PyPI

This package is set up for easy publishing to PyPI using `uv`. Here are the steps:

### Prerequisites

1. **Install uv** (if you haven't already):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Set up PyPI credentials**: 
   - Create an account on [PyPI](https://pypi.org/)
   - Generate an API token at https://pypi.org/manage/account/token/
   - Store it as an environment variable: `export UV_PUBLISH_TOKEN=pypi-...`

### Publishing Steps

1. **Test on TestPyPI first** (recommended):
   ```bash
   ./publish-test.sh
   ```
   This will build and upload to [TestPyPI](https://test.pypi.org/), where you can test the package safely.

2. **Publish to PyPI**:
   ```bash
   ./publish.sh
   ```
   This will build and upload to the real PyPI.

### Manual Publishing with uv

If you prefer to do it manually using the official uv workflow:

```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info/

# Build the package with uv
uv build

# Publish to TestPyPI (optional, for testing)
uv publish --index testpypi

# Publish to PyPI
uv publish
```

### Version Management

Use uv's built-in version management:

```bash
# Update to a specific version
uv version 1.0.0

# Bump version semantically
uv version --bump patch    # 0.1.0 -> 0.1.1
uv version --bump minor    # 0.1.0 -> 0.2.0
uv version --bump major    # 0.1.0 -> 1.0.0

# Preview changes without applying
uv version 2.0.0 --dry-run
```

The version will be automatically updated in both `pyproject.toml` and `latteries/__init__.py`.

### Package Structure

The package includes:
- Core API calling functionality
- Caching system
- Multi-provider support (OpenAI, Anthropic, etc.)
- Response viewer CLI tool (`latteries-viewer`)
- Example scripts and evaluation tools

### Testing Your Package

Test that the package works correctly after publishing:

```bash
# Test import without local project
uv run --with latteries --no-project -- python -c "import latteries; print(latteries.__version__)"

# If recently published, refresh cache
uv run --with latteries --no-project --refresh-package latteries -- python -c "import latteries"
```