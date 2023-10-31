
from sklearn.model_selection import KFold
from sklearn import metrics

def run_kfold(machine, data, target, number_of_splits, continuous):
	kfold_object = KFold(n_splits=number_of_splits)
	kfold_object.get_n_splits(data)

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
	  
		machine.fit(data_train, target_train)
		prediction = machine.predict(data_test)

		if (continuous==True):
			r2 = metrics.r2_score(target_test, prediction)
			print("R square score: ", r2)
			print("\n\n")
		else:
			accuracy_score = metrics.accuracy_score(target_test, prediction)
			print("Accuracy score: ", accuracy_score)

			confusion_matrix = metrics.confusion_matrix(target_test, prediction)
			print("Confusion martix: ", confusion_matrix)









