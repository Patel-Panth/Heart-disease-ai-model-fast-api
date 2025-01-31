# Heart Diseasec Model usingFast Api

## Heart-disease-ai-model-fast-api
This model is trained on the dataset of heart disease available on Kaggle: Heart Disease Dataset.

This uses the Random Forest classifier for training the dataset, which gives an accuracy of 100% on the test dataset using the parameters RandomForestClassifier(n_estimators=25, criterion='gini').

You can also try other models like Decision Tree, which was also giving the best performance on the given data.

By applying feature engineering, you can create and drop features (use a correlation matrix for feature engineering). 

For more accuracy, you can also apply sampling if the data is unbalanced. Additionally, apply different scaling methods to achieve better accuracy.

For hyperparameter tuning, you can visit the site Random Forest Classifier Documentation, where you can find parameters to experiment with for higher performance. You can use GridSearchCV, RandomizedSearchCV, and other methods for hyperparameter tuning.

For run the api you need the libraries Flask.Use the command to install the flask. ```'!pip install flask'``` here is the list of librays:
```
1)sklearn
2)joblib
3)pandas
4)model
```
