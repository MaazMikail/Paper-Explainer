.PHONY: install test clean

install:
	pip install -e .

test:
	pytest tests/

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete