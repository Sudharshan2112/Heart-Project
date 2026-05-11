from flask import Flask, request, render_template
import pandas as pd
from heart_model import train_model, predict_heart_disease

app = Flask(__name__)

# Load dataset and train model on startup
csv_file = 'Heart_Disease_Prediction.csv'  # Make sure this file is in the same directory
df = pd.read_csv(csv_file)

# Train the model
model, scaler = train_model(df)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from the form (these names must match the form inputs)
    user_input = {
        'Age': request.form['age'],
        'Sex': request.form['sex'],
        'Chest pain type': request.form['cp'],
        'BP': request.form['bp'],
        'Cholesterol': request.form['chol'],
        'FBS over 120': request.form['fbs'],
        'EKG results': request.form['ekg'],
        'Max HR': request.form['thalach'],
        'Exercise angina': request.form['exang'],
        'ST depression': request.form['oldpeak'],
        'Slope of ST': request.form['slope'],
        'Number of vessels fluro': request.form['ca'],
        'Thallium': request.form['thal']
    }

    # Get prediction from the model
    result = predict_heart_disease(model, scaler, user_input)

    # Return result to be displayed in the HTML
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
