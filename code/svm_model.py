import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  # Import SVM classifier
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import GridSearchCV
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
# Function to compute top-k accuracy
def compute_acc(all_beams, only_best_beam, top_k=[1, 3, 5]):
    n_top_k = len(top_k)
    total_hits = np.zeros(n_top_k)

    n_test_samples = len(only_best_beam)
    if len(all_beams) != n_test_samples:
        raise Exception('Number of predicted beams does not match number of labels.')

    for samp_idx in range(len(only_best_beam)):
        for k_idx in range(n_top_k):
            hit = np.any(all_beams[samp_idx, :top_k[k_idx]] == only_best_beam[samp_idx])
            total_hits[k_idx] += 1 if hit else 0

    return np.round(total_hits / len(only_best_beam), 4)

# Function to load scenario data
def load_scenario_data(scenario_number):
    base_path = f"..\\scenarios\\scenario_{scenario_number}"
    df = pd.read_csv(f"{base_path}\\DataSetEdited_with_aoarate_scenario{scenario_number}.csv", index_col=0)
    df_out = pd.read_csv(f"{base_path}\\true_beams_scenario{scenario_number}.csv", index_col=0)
    best_beams = pd.read_csv(f"{base_path}\\best_beams_scenario{scenario_number}.csv", index_col=0)
    powers = pd.read_csv(f"{base_path}\\powers_s{scenario_number}.csv", index_col=0)
    return df, df_out, best_beams, powers

# List of available scenarios
available_scenarios = [36, 37, 38, 39]

# User input to select scenarios
print("Available scenarios:", available_scenarios)
selected_scenarios = input("Enter the scenarios you want to include (comma-separated, e.g., 36,37): ")
selected_scenarios = [int(s.strip()) for s in selected_scenarios.split(',') if int(s.strip()) in available_scenarios]

# Load selected scenario data
dfs = []
dfs_out = []
best_beams_list = []
powers_list = []

for scenario in selected_scenarios:
    df, df_out, best_beams, powers = load_scenario_data(scenario)
    dfs.append(df)
    dfs_out.append(df_out)
    best_beams_list.append(best_beams)
    powers_list.append(powers)

# Concatenate data from selected scenarios
df = pd.concat(dfs, ignore_index=True)
df_out = pd.concat(dfs_out, ignore_index=True)
best_beams = pd.concat(best_beams_list, ignore_index=True)
powers = pd.concat(powers_list, ignore_index=True)

best_beams = best_beams.to_numpy()
powers = powers.to_numpy()
beams_powers = np.concatenate((best_beams, powers), axis=1)

# Split data into training and testing sets
X_Train, X_Test, Y_Train, Y_Test = train_test_split(df, beams_powers, test_size=0.7, random_state=0)
Y_Train_beams, Y_Train_powers = np.split(Y_Train, 2, axis=1)
Y_Test_beams, Y_Test_powers = np.split(Y_Test, 2, axis=1)
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df, best_beams, test_size=0.25, random_state=0)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the parameter grid for GridSearchCV
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

# Create a SVM classifier
svm_classifier = SVC()

# Perform GridSearchCV to find the best combination of hyperparameters
grid_search = GridSearchCV(estimator=svm_classifier, param_grid=param_grid, cv=5, verbose=2)

# Fit the grid search to the data
grid_search.fit(X_train_scaled, y_train[:, 0])  

# Get the best parameters found by GridSearchCV
best_params = grid_search.best_params_

# Create an SVM classifier with the best parameters
best_svm_classifier = SVC(**best_params)

# Train the best classifier on the full training data
best_svm_classifier.fit(X_train_scaled, y_train[:, 0])

# Evaluate the best classifier on the test data
best_svm_predictions = best_svm_classifier.predict(X_test_scaled)

top_k_accuracy_svm = compute_acc(y_test, best_svm_predictions, top_k=[1, 3, 5])
print("Top-k Accuracy (SVM):", top_k_accuracy_svm)


# Plot actual vs. predicted beam indices
plt.figure(figsize=(8, 6))
plt.scatter(Y_Test[:, 0], best_svm_predictions, color='blue', label='Predicted', alpha=0.3)
plt.plot([min(Y_Test[:, 0]), max(Y_Test[:, 0])], [min(Y_Test[:, 0]), max(Y_Test[:, 0])], color='dimgray', linestyle='--', label='Ideal')
plt.title('Actual vs. Predicted Beam Indices (SVM)')
plt.xlabel('Actual Beam Index')
plt.ylabel('Predicted Beam Index')
plt.legend()
plt.grid(True)
plt.show()