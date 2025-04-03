import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ✅ Get absolute path of the script directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# ✅ Use absolute paths
csv_path = os.path.join(base_dir, "hiring_data.csv")
model_path = os.path.join(base_dir, "models", "model.pkl")
scaler_path = os.path.join(base_dir, "models", "scaler.pkl")

# ✅ Debugging: Print paths to check correctness
print("CSV Path:", csv_path)
print("Model Save Path:", model_path)

# ✅ Load dataset
df = pd.read_csv(csv_path)

# ✅ Define features and target
x = df[['SSLC Percentage', 'PU_percentage', 'UG_CGPA', 'Quants', 'LogicalReasoning',
        'Verbal', 'Programming', 'Communication', 'Experience']]
y = df["result"]  # 1 for Hired, 0 for Not Hired

# ✅ Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# ✅ Scale the features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# ✅ Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train_scaled, y_train)

# ✅ Evaluate model
y_pred = model.predict(x_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# ✅ Save model and scaler
os.makedirs(os.path.dirname(model_path), exist_ok=True)  # Create 'models' folder if not exists
with open(model_path, "wb") as f:
    pickle.dump(model, f)

with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)

print("Model training complete! Saved as 'models/model.pkl'")
