PYSRC = src

# Python
checkformat:
	isort --check-only ${PYSRC}
	black --check ${PYSRC}

format:
	isort ${PYSRC}
	black ${PYSRC}

lint: ruff mypy

ruff:
	ruff check ${PYSRC}

mypy:
	mypy ${PYSRC} --install-types --non-interactive --no-namespace-packages

test:
	pytest ${PYSRC} -vv

commit:
	cz commit

.PHONY: checkformat format lint ruff mypy pytest commit
