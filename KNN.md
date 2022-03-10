# explore KNN in ship risk predictions and machine learning training model brief explanation

# Brief Explanantions of Data：

- **Application to tackle Maritime issues**
  
  We collect some data samples from maritime industry, then we want to use KNN to predict which level of risk the given vessel is holding, therefore, it will provide more information for PSC officer to keep watch the specific ships and check further potential deficiences of items onboard, like fire protection aplliances, life saving equipments.

- **Source:** TokyoMOU DataBase

- **Features** 
  
  ![data_explanations.png](D:\Devp\MachineLearning\ML%20Algorithm\img\data_explanations.png)
  
  1.In order to explain KNN algorithm more easily, I only choose two features-Flag and Deficiencies, because these two pose the most affect on that the ship burden the high/standar or low risk. However, if we take it into the real practice, tonnes of features need to be extracted.
  
  2.Because of flag is not the ordinal, so I pose"get_dummies" method in pandas to handle the problem.
  
  **3.Codes**
  
  ```python
  mport pandas as pd
  import numpy as n
  
  df = pd.read_excel("C:/Users/guoha/Desktop/PSC.xlsx") #read into the data
  
  #choose the nominated columns and change the column name
  df = df[["Flag","Deficiencies","Ship Risk Profile at the time of inspection"]]
  df.rename(columns={"Ship Risk Profile at the time of inspection":"Risk"}, inplace=True)
  
  #drop the null "risk" rows
  df.dropna(axis=0,how="any",inplace=True)
  
  #0,1,2 stands for the risk level(from low to high, it exists order)
  df = df.replace({'Low Risk Ship':0, 'Standard Risk Ship':1,'High Risk Ship':2 })
  
  df_1 = df[["Deficiencies","Risk"]]
  df_2 = pd.get_dummies(df.Flag)
  raw_result = pd.concat([df_2,df_1],axis=1)
  
  # split data into X(attributes) and Y(predictions)
  raw_data_X = raw_result.drop(["Risk"], axis=1)
  raw_data_Y = raw_result.Risk
  
  X_train = np.array(raw_data_X)
  y_train = np.array(raw_data_Y)
  # X_train.shape, y_train.shape
  
  # I use the first low as example to predict
  x = np.array(raw_data_X.iloc[0])
  
  #calculate the distances between x(example of preidctions) with all data point by Euler's formula
  from math import sqrt
  distances = [sqrt(np.sum(x_train-x)**2) for x_train in X_train]
  
  #order the distances
  orders = np.argsort(distances)
  # we nominated k=10,
  
  prediction_list = [y_train[n] for n in orders[:k]]
  # result of prediction_list =[1, 1, 1, 1, 1, 0, 0, 0, 2, 0, 0, 1]
  
  from collections import Counter
  predict_result = Counter(prediction_list).most_common(1)[0][0]
  print(f"The prediction of risk level is {predict_result}")
  print("Notes: 0=Low Risk; 1=Standard Risk; 2=High Risk")
  
  """
  we can predict 10 vessels or more:
  
  predict_object=np.array(raw_data_X.iloc[:10])
  from math import sqrt
  dist_cum = []
  for x in predict_object:
      distances = [sqrt(np.sum(x_train-x)**2) for x_train in X_train]
      dist_cum.append(distances)
  k=10
  orders = []
  for distance in dist_cum:
      orders.append(np.argsort(distance)
  
  prediction_list = [])
  for order in orders:
      prediction_list.append([y_train[n] for n in order[:k]])
  
  from collections import Counter
  predict_result = []
  for num in prediction_list:
      predict_result.append(Counter(num).most_common(1)[0][0])
  print(f"The prediction of risk level is {predict_result}")
  print("Notes: 0=Low Risk; 1=Standard Risk; 2=High Risk")
  
  ###python:
  The prediction of risk level is [1, 1, 1, 1, 1, 1, 1, 2, 1, 2]
  Notes: 0=Low Risk; 1=Standard Risk; 2=High Risk
  -------------------------------------------------------------------
  ```
  
  Regarding Above, then we use KNN from SKlearn:
  
  from sklearn.neighbors import KNeighborsClassifier
  KNN_classifier = KNeighborsClassifier(n_neighbors=10)
  KNN_classifier.fit(X_train,y_train)
  
  y_predict = KNN_classifier.predict(predict_object)
  
  ###array([0, 1, 1, 1, 1, 1, 1, 1, 0, 2], dtype=int64)==>seems prdiction result is better!
  
  4.In pratice, how we value the prediction, it's accurate? or we have enough confidence to say 'that's the perfect model' ? Shall we use it for our new dataset predictions or not? Besieds, we shall ponder, time to time, if prediction is wrong: mistakes 0 to 1 or to 2, it seems to be accepted by PSC officer, however, 2 to 0 or 1, which will be a disaster, right? Because we miss the high risk vessels!

# Valuation of the model

accuracy_score:

```python
import numpy as np

def accuracy_score(y_true, y_predict):
    '''calculation accuracy rate between y_true and y_predict'''
    assert y_true.shape[0] == y_predict.shape[0], \
        "the size of y_true must be equal to the size of y_predict"

    return sum(y_true == y_predict) / len(y_true)
```

# Feature Scaling

- StandardScaler

- MinMaxScaler

please see the documentation, but we can do it via Pipeline, it's more easy and  it could be associated with feature engineering.

# Tuning parameters

- search the best K

```python
best_score = 0.0
best_k = -1
for k in range(1, 11):
    knn_clf = KNeighborsClassifier(n_neighbors=k)
    knn_clf.fit(X_train, y_train)
    score = knn_clf.score(X_test, y_test)
    if score > best_score:
        best_k = k
        best_score = score

print("best_k =", best_k)
print("best_score =", best_score)
```

- search the weights

```python
best_score = 0.0
best_k = -1
best_method = ""
for method in ["uniform", "distance"]:
    for k in range(1, 11):
        knn_clf = KNeighborsClassifier(n_neighbors=k, weights=method)
        knn_clf.fit(X_train, y_train)
        score = knn_clf.score(X_test, y_test)
        if score > best_score:
            best_k = k
            best_score = score
            best_method = method

print("best_method =", best_method)
print("best_k =", best_k)
print("best_score =", best_score)
```

- search the best distance calculation format

```python
best_score = 0.0
best_k = -1
best_p = -1

for k in range(1, 11):
    for p in range(1, 6):
        knn_clf = KNeighborsClassifier(n_neighbors=k, weights="distance", p=p)
        knn_clf.fit(X_train, y_train)
        score = knn_clf.score(X_test, y_test)
        if score > best_score:
            best_k = k
            best_p = p
            best_score = score

print("best_k =", best_k)
print("best_p =", best_p)
print("best_score =", best_score)
```

- p=1：Manhattan distance(曼哈顿距离)
- p=2：Euclidean distance（欧氏距离）
- p=infinite：Chebyshev distance（切比雪夫距离）

Therefore, if we want to use one model accurately, we shall get the best knowledge of its parameters to make sure which one is the best, in scikit learn, we have two main method to gid out the best parameter to make our model the best, which are grid search and radom search.

- Grid Search

```python
param_grid = [
    {
        'weights': ['uniform'], 
        'n_neighbors': [i for i in range(1, 11)]
    },
    {
        'weights': ['distance'],
        'n_neighbors': [i for i in range(1, 11)], 
        'p': [i for i in range(1, 6)]
    }
]

knn_clf = KNeighborsClassifier()

from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(knn_clf, param_grid)

grid_search.fit(X_train, y_train)
```
