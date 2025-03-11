import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Function to train the model
def train_model(df):
    # Preprocess the dataset
    df['Heart Disease'] = df['Heart Disease'].map({'Presence': 1, 'Absence': 0})  # Encode target variable

    # Features (X) and target (y)
    X = df.drop('Heart Disease', axis=1)
    y = df['Heart Disease']

    # Split the data into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    return model, scaler  # Return the trained model and scaler for predictions

# Function to make predictions based on user input
def predict_heart_disease(model, scaler, user_input):
    # Convert input to DataFrame for prediction
    input_df = pd.DataFrame([user_input])

    # Scale the input data using the same scaler used during training
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_scaled)[0]

    # Return result based on prediction
    return 'Heart Disease Detected' if prediction == 1 else 'No Heart Disease'
