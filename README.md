# COVID-19-Crowd-Counting

Link to project report: https://docs.google.com/document/d/1urggk9yMcrpUDIhJftpAdjG-CM6nM6o1q43KzL9jgng/edit?usp=sharing

## Requirements
To install and make sure you have all requirements necessary to run our code, you can run `pip install -r requirements.txt` from the root directory to verify necessary packages and install missing ones.

## Training Models
There are 4 scripts to train our models:
- Baseline Classification: `train_baseline_classification.py`
- VGG16 Classification Transfer Learning: `train_vgg16_classification_script.py`
- ResNet50 Density Map Generation: `train_resnet50_script.py`
- VGG16 Density Map Generation: `train_vgg16_script.py`

## Evaluating Models:
Models can be evaluated on their saved losses using `evaluate_classification_training.py` and `evaluate_regression_training.py`. Models can also be evaluated on training/validation/testing data with `evaluate_classification_model.py --model='model_name'` and `evaluate_classification_regression_model.py --model='model_name'`
