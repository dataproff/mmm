.PHONY: help setup dbt-run dbt-test train-model run-app docker-build deploy clean

help:
	@echo "MMM Project Commands:"
	@echo "  make setup        - Install all dependencies"
	@echo "  make dbt-run      - Run DBT models"
	@echo "  make dbt-test     - Test DBT models"
	@echo "  make train-model  - Train Robyn MMM model"
	@echo "  make run-app      - Run Streamlit app"
	@echo "  make docker-build - Build Docker image for training"
	@echo "  make clean        - Clean generated files"

setup:
	@echo "Installing DBT dependencies..."
	cd dbt_project && dbt deps
	@echo "Installing Python dependencies..."
	pip install -r robyn_training/requirements.txt
	pip install -r streamlit_app/requirements.txt
	@echo "Setup complete!"

dbt-run:
	@echo "Running DBT models..."
	cd dbt_project && dbt seed && dbt run

dbt-test:
	@echo "Testing DBT models..."
	cd dbt_project && dbt test

train-model:
	@echo "Training Robyn MMM model..."
	cd robyn_training && python train_model.py

run-app:
	@echo "Starting Streamlit app..."
	cd streamlit_app && streamlit run app.py

docker-build:
	@echo "Building Docker image..."
	cd robyn_training && docker build -t mmm-robyn-trainer .

deploy-app:
	@echo "Deploying Streamlit app to Cloud Run..."
	gcloud builds submit --tag gcr.io/$(GCP_PROJECT_ID)/mmm-app streamlit_app/
	gcloud run deploy mmm-app \
		--image gcr.io/$(GCP_PROJECT_ID)/mmm-app \
		--platform managed \
		--region us-central1 \
		--allow-unauthenticated

clean:
	@echo "Cleaning generated files..."
	rm -rf dbt_project/target
	rm -rf dbt_project/dbt_packages
	rm -rf robyn_training/models/*
	rm -rf robyn_training/plots/*
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "Clean complete!"
