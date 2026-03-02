import pandas as pd
df = pd.read_csv('engine_data.csv')
df.columns = [col.lower().replace(' ', '_') for col in df.columns]
print("DataFrame loaded and columns cleaned. First 5 rows:")
print(df.head())

from sklearn.model_selection import train_test_split

X = df.drop('engine_condition', axis=1)
y = df['engine_condition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Features (X) and target (y) separated.")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

from sklearn.ensemble import RandomForestClassifier

# Initialize the RandomForestClassifier with class_weight='balanced'
model = RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100)

# Train the model
model.fit(X_train, y_train)

print("RandomForestClassifier model trained successfully with class_weight='balanced'.")

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy:.4f}")

# Generate and print the classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Generate and print the confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

import matplotlib.pyplot as plt
import pandas as pd

# Get feature importances
feature_importances = model.feature_importances_

# Create a Series for better handling and sort them
importances_df = pd.Series(feature_importances, index=X.columns).sort_values(ascending=False)

# Plotting feature importances
plt.figure(figsize=(10, 6))
importances_df.plot(kind='bar')
plt.title('Feature Importances for Engine Condition Prediction')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

print("Feature importances calculated and visualized.")

import joblib

# Save the trained model
joblib.dump(model, 'engine_model.pkl')

print("Trained RandomForestClassifier model saved as 'engine_model.pkl'.")

import pandas as pd

def predict_engine_health(new_sensor_data_df):
    """
    Predicts engine failure probability and classifies engine health.

    Args:
        new_sensor_data_df (pd.DataFrame): DataFrame with new sensor input values.
                                           Column names must match training features.

    Returns:
        tuple: A tuple containing:
               - float: Predicted probability of engine failure (class 1).
               - str: Classified engine health ('GOOD', 'WARNING', 'CRITICAL').
    """
    # Ensure the input DataFrame has the same column order as X.columns used for training
    if not list(new_sensor_data_df.columns) == list(X.columns):
        # Attempt to reorder columns if they are present, otherwise raise an error
        try:
            new_sensor_data_df = new_sensor_data_df[X.columns]
        except KeyError as e:
            raise ValueError(f"Missing expected feature columns in input data: {e}")

    # Predict probabilities for each class
    probabilities = model.predict_proba(new_sensor_data_df)

    # Get the probability of class 1 (engine failure)
    failure_probability = probabilities[0, 1] # Assuming single row input for simplicity here

    # Classify engine health based on probability thresholds
    if failure_probability < 0.3:
        engine_health = 'GOOD'
    elif 0.3 <= failure_probability <= 0.7:
        engine_health = 'WARNING'
    else:
        engine_health = 'CRITICAL'

    return failure_probability, engine_health

print("Function 'predict_engine_health' defined successfully.")

safe_ranges = {
    'engine_rpm': (500, 1500), # Typical RPM range
    'lub_oil_pressure': (2.0, 6.0), # Normal operating pressure in bar
    'fuel_pressure': (10.0, 20.0), # Normal fuel pressure in bar
    'coolant_pressure': (1.0, 4.0), # Normal coolant pressure in bar
    'lub_oil_temp': (60.0, 95.0), # Normal lubrication oil temperature in Celsius
    'coolant_temp': (75.0, 90.0) # Normal coolant temperature in Celsius
}

print("Safe ranges for sensor parameters defined.")