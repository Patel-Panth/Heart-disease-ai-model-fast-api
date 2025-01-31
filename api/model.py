
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import accuracy_score,mean_absolute_error
from sklearn.base import BaseEstimator, TransformerMixin

class PreProcessing(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self  
    
    def transform(self, data):
        categorical_val = []
        continuous_val = []
        for column in data.columns:
            if len(data[column].unique()) <= 10:
                categorical_val.append(column)
            else:
                continuous_val.append(column)

        def health_status(row):
            if row['oldpeak'] > 1.17 and row['slope'] < 2 and row['maximum heart rate achieved'] > 149 and row['fasting blood sugar'] == 1 and row['trestbps'] > 131:
                return 1
            else:
                return 0

        data['health'] = data.apply(health_status, axis=1)
        data = data.drop(['sex', 'age', 'fasting blood sugar', 'serum cholestoral'], axis=1)
        return data


class Scaling(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, X, y=None):       
        continous_val = ['trestbps', 'maximum heart rate achieved', 'oldpeak']
        X[continous_val] = self.scaler.fit_transform(X[continous_val])
        return X

    def transform(self, data):      
        data[['trestbps', 'maximum heart rate achieved', 'oldpeak']] = self.scaler.transform(
            data[['trestbps', 'maximum heart rate achieved', 'oldpeak']])
        return data

    def get_params(self, deep=True):
        return {"scaler": self.scaler}

class ModelTrainer(BaseEstimator):
    def __init__(self):
        self.accuracy = None
        self.mse = None

    def fit(self, X, y):
        self.model = RandomForestClassifier(n_estimators=25, criterion='gini')
        self.model.fit(X, y)
        # self.accuracy = self.model(X,y)
        # self.mse = 1- self.accuracy
        joblib.dump(self.model, 'model.pkl')
        return self

    def predict(self, X):
        pred = self.model.predict(X)
        
        return pred
    
    def get_matrices(self):
        return {'accuracy' : self.accuracy,'mean absolute erroe' : self.mse}
    
    
        
