import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import data_processing as dp
data_train=dp.X_train

# Bibliothèques de Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from catboost import CatBoostClassifier, Pool
# Chargement des données
