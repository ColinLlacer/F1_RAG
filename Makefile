.PHONY: check

check:
	poetry run flake8 .
	poetry run pytest 