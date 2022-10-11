# Environmental Sound Classifier

Implements an environmental sound classifier using the following:

- Code quality assurance with pre-commit hooks, GitHub Actions, and pytest
- Model implementation in PyTorch
- Model training and basic tracking and metrics via PyTorch Lightning and torchmetrics
- Experiment tracking, hyperparameter tuning, and model versioning with W&B
- Model packaging in TorchScript
- Predictor backend containerization via Docker and deployment as a microservice on AWS Lambda
- Pure Python frontend web application in Gradio, tunneled with Ngrok. Running on an EC2 instance.

## Data

The model was trained on the [ESC-50: Dataset for Environmental Sound Classification](https://github.com/karolpiczak/ESC-50).
PyTorch Lightning Dataset and Datamodule implemented for this particular dataset.

## Training

The model is a pretrained resnet, modified and fine-tuned for the task. Nothing fancy here.
The audio was converted to Mel Spectrogram before being fed to the model.
