import pandas
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn import metrics

dataset = pandas.read_csv("dataset.csv")

#shuffle
dataset = dataset.sample(frac=1)

print(dataset)

#separate target and data
target = dataset.iloc[:,0].values
data = dataset.iloc[:,3:9].values


#every part of the testing portion including Kfold should be in template we create, n splits can change though
kfold_object = KFold(n_splits=4)
kfold_object.get_n_splits(data)

# print(kfold_object)

i=0
for train_index, test_index in kfold_object.split(data):
  i=i+1
  print("Round:", str(i))
  print("Training index: ")
  print(train_index)
  print("Testing index: ")
  print(test_index)
  
  data_train = data[train_index]
  target_train = target[train_index]
  data_test = data[test_index]
  target_test = target[test_index]
  
  #shouldn't include this part because we might want to use diff model in future
  machine = linear_model.LinearRegression()
  machine.fit(data_train, target_train)
  
  prediction = machine.predict(data_test)
  
  #shouldn't include r2 because we might want to use accuracy etc, we will make template know which one to use
  r2 = metrics.r2_score(target_test, prediction)
  print("R square score: ", r2)
  print("\n\n")




  