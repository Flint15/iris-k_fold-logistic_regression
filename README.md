# Getting started

## Prerequisites

* scikit_learn
* numpy

## Instalation

`pip install scikit-learn`

`pip install numpy`

# Using

### Import Necessary libraries and load the data
```python
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import numpy as np

#Load dataset
iris = load_iris()
X, y = iris.data, iris.target
```

### Create a list for count accuracy

```python
accuracies = []
```

### Initialize a model and KFold

```python
model = LogisticRegression(max_iter=200)
kf = KFold(n_splits=5, shuffle=True, random_state=0)
```

### Perform KFold cross-validation

<i>Note that we write `kf.split(X)` we use feuture data - X, for create indices </i>
```python
for train_index, test_index in kf.split(X):
  
  #Split data into training and testing sets by using indices
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]

  #Train and make a prediction
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)

  #Calculate accuracy
  accuracy = accuracy_score(y_test, y_pred)
  accuracies.append(accuracy)
```
