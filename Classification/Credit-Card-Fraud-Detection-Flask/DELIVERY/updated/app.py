from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.exceptions import NotFittedError
import joblib
from datetime import datetime

app = Flask(__name__)

# Load your trained model
model = joblib.load('fraud_detection_model.pkl')

# Load encoders for categorical columns
encoders = {
    'merchant': joblib.load('encoders/merchant_encoder.pkl'),
    'category': joblib.load('encoders/category_encoder.pkl'),
    'first': joblib.load('encoders/first_encoder.pkl'),
    'last': joblib.load('encoders/last_encoder.pkl'),
    'gender': joblib.load('encoders/gender_encoder.pkl'),
    'street': joblib.load('encoders/street_encoder.pkl'),
    'city': joblib.load('encoders/city_encoder.pkl'),
    'state': joblib.load('encoders/state_encoder.pkl'),
    'job': joblib.load('encoders/job_encoder.pkl')
}

def preprocess_features(features_df):
    # Convert 'trans_date_trans_time' and 'dob' to datetime and extract features
    if 'trans_date_trans_time' in features_df:
        features_df['trans_date_trans_time'] = pd.to_datetime(features_df['trans_date_trans_time'])
        features_df['year'] = features_df['trans_date_trans_time'].dt.year
        features_df['month'] = features_df['trans_date_trans_time'].dt.month
        features_df['day'] = features_df['trans_date_trans_time'].dt.day
        features_df['hour'] = features_df['trans_date_trans_time'].dt.hour
        features_df['weekday'] = features_df['trans_date_trans_time'].dt.weekday
        features_df.drop('trans_date_trans_time', axis=1, inplace=True)

    if 'dob' in features_df:
        features_df['dob'] = pd.to_datetime(features_df['dob'])
        current_year = datetime.now().year
        features_df['age'] = current_year - features_df['dob'].dt.year
        features_df.drop('dob', axis=1, inplace=True)
    
    # Encode categorical features using pre-loaded LabelEncoders
    try:
        for col, encoder in encoders.items():
            if col in features_df.columns:
                features_df[col] = encoder.transform(features_df[col])
    except ValueError as e:
        # This error typically happens if the label is unseen
        raise ValueError(f"Invalid input for {col}: {features_df[col].iloc[0]}")
            
    # Ensure features are in the correct order as the model expects
    feature_order = ['cc_num', 'merchant', 'category', 'amt', 'first', 'last', 'gender',
                     'street', 'city', 'state', 'zip', 'lat', 'long', 'city_pop', 'job',
                     'unix_time', 'year', 'month', 'day', 'hour', 'weekday', 'age']
    features_df = features_df[feature_order]

    return features_df

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()
        features_df = pd.DataFrame([data])
        
        if not all(features_df.iloc[0]):  # Checks if any field in the first row is empty
            raise ValueError("All fields must be filled out.")
        
        processed_features = preprocess_features(features_df)
        prediction = model.predict(processed_features)
        prediction_proba = model.predict_proba(processed_features)
        fraud_probability = prediction_proba[0][1]
        return render_template('result.html', prediction=int(prediction[0]), probability=fraud_probability)
    except ValueError as ve:
        # Redirect back to form with error message specific to the problem
        print(ve)
        return render_template('index.html', data=request.form, error=str(ve))
    except Exception as e:
        # General error handling
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
