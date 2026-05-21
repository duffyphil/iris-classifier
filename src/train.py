#Load the data
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data  #150,4  #sepal length, sepal width, petal length, petal width
y = iris.target #150,  #Setosa(0), Versicolor(1), Virginica(2)

#Split into training  and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)  #80% train, 20% test

#Choose, initialise and train a Decision Tree model
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

#Predict species of flowers in our test set
y_pred = model.predict(X_test)

#Accuracy metric
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Create outputs folder if it doesn't exist
import os
os.makedirs("outputs", exist_ok=True)

#Save the matrix to an image file
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png") 
