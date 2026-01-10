install:
	@echo "Installing dependencies..."
	@pip install -r requirements.txt

format:
	@echo "Formatting code..."
	@ruff format .

check:
	@echo "Checking code..."
	@ruff check .

fix-lint:
	@echo "Fixing lint..."
	@ruff check . --fix

isort:
	@echo "Sorting imports..."
	@isort .

isort-check:
	@echo "Checking imports..."
	@isort . --check-only

format-all:
	@echo "Formatting code..."
	@ruff format .
	@isort .
	@ruff check . --fix

check-all:
	@echo "Checking code..."
	@ruff check .
	@isort . --check-only

verify-data:
	@echo "Verifying data..."
	@python scripts/verify_data.py

clean:
	@echo "Cleaning up..."
	@find . -type d -name "__pycache__" -exec rm -rf {} +