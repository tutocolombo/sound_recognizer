# Arcane incantation to print all the other targets, from https://stackoverflow.com/a/26339924
help:
	@$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$'

# Install exact Python and CUDA versions
conda-update:
	conda env update --prune -f environment.yml
	echo "!!!RUN THE conda activate COMMAND ABOVE RIGHT NOW!!!"

# Compile and install exact pip packages
pip-tools:
	pip install pip-tools==6.9.0 setuptools==63.1.0
	pip-compile requirements/prod.in && pip-compile requirements/dev.in
	pip-sync requirements/prod.txt requirements/dev.txt

# Compile and install the requirements for local linting (optional)
pip-tools-lint:
	pip install pip-tools==6.9.0 setuptools==63.1.0
	pip-compile requirements/prod.in && pip-compile requirements/dev.in && pip-compile requirements/dev-lint.in
	pip-sync requirements/prod.txt requirements/dev.txt requirements/dev-lint.txt

# Bump versions of transitive dependencies
pip-tools-upgrade:
	pip install pip-tools==6.9.0 setuptools==63.1.0
	pip-compile --upgrade requirements/prod.in && pip-compile --upgrade requirements/dev.in && pip-compile --upgrade requirements/dev-lint.in

# Build Docker images
build-api-docker-image:
    docker build -t api_serverless -f api_serverless/Dockerfile .

build-app-docker-image:
    docker build -t app_gradio -f app_gradio/Dockerfile .