# Sports Ball Classification

This repository contains a deep learning (DL)-based artificial intelligence (AI) image classification model training to classify balls used in different sports types. The AI model used for the classification task is RexNet ([paper](https://arxiv.org/pdf/2007.00992.pdf) and [code](https://github.com/clovaai/rexnet)) and the dataset for training is [Sports Balls Dataset](https://www.kaggle.com/datasets/samuelcortinhas/sports-balls-multiclass-image-classification). The project in [Kaggle](https://www.kaggle.com/) can be found [here](https://www.kaggle.com/code/killa92/sportballs-visualization-classification-pytorch).

# Manual on how to use the repo:

1. Clone the repo to your local machine using terminal via the following script:

```python
git clone https://github.com/bekhzod-olimov/SportsBallClassification.git
```

2. Create conda environment from yml file using the following script:
```python
conda env create -f environment.yml
```
Then activate the environment using the following command:
```python
conda activate speed
```

3. Data Visualization

![image](https://github.com/bekhzod-olimov/SportsBallClassification/assets/50166164/cc6e4e03-3487-41c9-b6d5-94b59d4630d4)

4. Train the AI model using the following script:
```python
python main.py --root PATH_TO_THE_DATA --batch_size = 64 device = "cuda:0"
```
The training parameters can be changed using the following information:

![image](https://github.com/bekhzod-olimov/SportsBallClassification/assets/50166164/41a3b906-9ee0-4a12-b4c0-3262841c7861)

The training process progress:

![image](https://github.com/bekhzod-olimov/JellyfishClassifier/assets/50166164/ca68a5af-bef1-4b41-9412-648c2f4942c3)

5. Learning curves:
   
Use [DrawLearningCurves](https://github.com/bekhzod-olimov/JellyfishClassifier/blob/80393cea3cdf497533f915d88481a3513b6cbcf7/main.py#L56C6-L56C6) class to plot and save learning curves.

* Train and validation loss curves:
  
![loss_learning_curves](https://github.com/bekhzod-olimov/SportsBallClassification/assets/50166164/16db0e9d-dd69-40c4-83d9-199038e3b542)

* Train and validation accuracy curves:
  
![acc_learning_curves](https://github.com/bekhzod-olimov/SportsBallClassification/assets/50166164/fed231d0-4c8b-4398-ae3c-2d59a4421d51)

6. Conduct inference using the trained model:
```python
python inference.py --root PATH_TO_THE_DATA --batch_size = 64 device = "cuda:0"
```

The inference progress:

![image](https://github.com/bekhzod-olimov/JellyfishClassifier/assets/50166164/8fee0d75-c43c-4b85-9fcd-9a285a4cdf4a)

7. Inference Results (Predictions):

![brain_preds](https://github.com/bekhzod-olimov/SportsBallClassification/assets/50166164/7a8ffd57-4f66-4383-a5f5-06d032dbc27f)
