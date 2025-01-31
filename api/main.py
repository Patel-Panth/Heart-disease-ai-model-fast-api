from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from model import PreProcessing,Scaling,ModelTrainer


app = Flask(__name__)


pipeline = Pipeline([
    ('preprocessing', PreProcessing()),  
    ('scaling', Scaling()),             
    ('trainer', ModelTrainer())          
])



@app.route('/')
def home():
    return "Welcome to the Heart Disease Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        try:
            model = joblib.load('model.pkl')
            
        except:
            model = None
            
        input_data = pd.DataFrame([data])

        # model = joblib.load('model.pkl')
        processed_data = pipeline.named_steps['preprocessing'].transform(input_data)
        scaled_data = pipeline.named_steps['scaling'].transform(processed_data)

        prediction = model.predict(scaled_data)

        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/train', methods=['POST'])
def train():
    try:
        data = pd.read_csv('D:\mldl\heart disease\heart.csv')
        df = pd.DataFrame(data)

        X = df.drop('target', axis=1) 
        y = df['target']

        processed_data = pipeline.named_steps['preprocessing'].transform(X)
        scaled_data = pipeline.named_steps['scaling'].fit_transform(processed_data)

        pipeline.named_steps['trainer'].fit(scaled_data, y)

        joblib.dump(pipeline.named_steps['trainer'].model, 'model.pkl')
        joblib.dump(pipeline.named_steps['scaling'].scaler, 'scaler.pkl')

        return jsonify({'message': 'Model trained and saved successfully!'})

    except Exception as e:
        return jsonify({'error': str(e)})


@app.route("/traning_score",methods = ['GET'])
def get_metrices():
    metrices = pipeline.named_steps['trainer'].get_matrices()

    return metrices

if __name__ == '__main__':
    app.run(debug=True)
