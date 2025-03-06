import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# Define the dataset path (Replace with your actual dataset path)
DATASET_PATH = "path/to/dataset.csv"
OUTPUT_RESULTS_PATH = "output_results.csv"

# Load dataset
print("Loading dataset...")
df = pd.read_csv(DATASET_PATH)

# Define target variable (Y) and independent variables (X)
Y = df["Class"].astype(int)  # Convert target variable to integer
X = df.drop(labels=["Class"], axis=1)  # Drop the label column

# Split data into training (80%) and testing (20%) sets
print("Splitting dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

# Standardizing features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train Random Forest model
print("Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=1200, 
    min_samples_split=4, 
    min_samples_leaf=10, 
    max_features='sqrt', 
    max_depth=10, 
    criterion='gini',
    random_state=42
)

start_time = time.time()
model.fit(X_train, y_train)
end_time = time.time()
training_time = end_time - start_time

# Make predictions
print("Evaluating model on test data...")
predictions = model.predict(X_test)

# Compute evaluation metrics
results = {
    "auc_roc": roc_auc_score(y_test, predictions),
    "accuracy": accuracy_score(y_test, predictions),
    "precision": precision_score(y_test, predictions),
    "recall": recall_score(y_test, predictions),
    "f1_score": f1_score(y_test, predictions),
    "training_time_sec": training_time
}

# Save results to CSV
df_results = pd.DataFrame([results])
df_results.to_csv(OUTPUT_RESULTS_PATH, index=False)
print(f"Analysis completed. Results saved to {OUTPUT_RESULTS_PATH}")
