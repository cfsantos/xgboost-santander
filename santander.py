from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot
import pandas as pd
import numpy as np

# carrega dados
X = pd.read_csv(filepath_or_buffer="train.csv",
                          index_col=0, sep=',')
y = X["TARGET"]
X = X.drop(labels="TARGET", axis=1)

ratio = float(np.sum(y == 1)) / np.sum(y==0)

# classifica os dados
model = XGBClassifier(max_depth = 5,
                learning_rate=0.05, 
                n_estimators=200,
                scale_pos_weight = ratio)
model.fit(X, y)


# Gera grafico com importancias
plot_importance(model)
pyplot.show()
