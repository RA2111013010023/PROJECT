import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter

# Function to load CSV data and extract features
def load_data_from_csv(file_path):
    df = pd.read_csv(file_path)
    
    # Extract relevant features
    features = df[['Left_Knee_Angle', 'Right_Knee_Angle', 'Head_Midline', 'Chin_Tuck',
                   'Left_Shoulder_Lifted', 'Right_Shoulder_Lifted', 'Left_Ankle_Lifted', 'Right_Ankle_Lifted']].values
    
    return features, df

# Function to generate labels based on supine lower criteria
def generate_labels(features, thresholds):
    labels = []
    
    # Convert features to a DataFrame for easier column indexing
    df = pd.DataFrame(features, columns=['Left_Knee_Angle', 'Right_Knee_Angle', 'Head_Midline', 'Chin_Tuck',
                                         'Left_Shoulder_Lifted', 'Right_Shoulder_Lifted', 'Left_Ankle_Lifted', 'Right_Ankle_Lifted'])
    
    # Define supine lower criteria
    is_supine_lower = df['Head_Midline']  # Ensure this correctly represents supine lower
    left_hip_knee_angle = df['Left_Knee_Angle']
    right_hip_knee_angle = df['Right_Knee_Angle']

    # Loop through each row and assign labels
    for i in range(len(df)):
        if not is_supine_lower[i] or is_supine_lower[i] < thresholds['head_midline']:
            labels.append(0)  # Thighs and legs on surface
        else:
            hip_flexed = (left_hip_knee_angle[i] < thresholds['left_knee_angle']) and (right_hip_knee_angle[i] < thresholds['right_knee_angle'])
            labels.append(2 if hip_flexed else 1)  # 2 if flexed, 1 if not

    return labels

# Function to calculate thresholds from CSV files
def calculate_thresholds(csv_folder_path):
    all_file_names = [file_name for file_name in os.listdir(csv_folder_path) if file_name.endswith(".csv")]
    all_data = pd.DataFrame()

    for file_name in all_file_names:
        file_path = os.path.join(csv_folder_path, file_name)
        _, df = load_data_from_csv(file_path)
        all_data = pd.concat([all_data, df], ignore_index=True)

    thresholds = {
        'left_knee_angle': all_data['Left_Knee_Angle'].mean() + all_data['Left_Knee_Angle'].std(),
        'right_knee_angle': all_data['Right_Knee_Angle'].mean() + all_data['Right_Knee_Angle'].std(),
        'head_midline': all_data['Head_Midline'].mean() - all_data['Head_Midline'].std(),
        'chin_tuck': all_data['Chin_Tuck'].mean() - all_data['Chin_Tuck'].std(),
        'left_shoulder_lifted': all_data['Left_Shoulder_Lifted'].mean() - all_data['Left_Shoulder_Lifted'].std(),
        'right_shoulder_lifted': all_data['Right_Shoulder_Lifted'].mean() - all_data['Right_Shoulder_Lifted'].std(),
        'left_ankle_lifted': all_data['Left_Ankle_Lifted'].mean() - all_data['Left_Ankle_Lifted'].std(),
        'right_ankle_lifted': all_data['Right_Ankle_Lifted'].mean() - all_data['Right_Ankle_Lifted'].std(),
    }

    return thresholds

# Function to train the model
def train_model(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    return clf

# Function to evaluate the model on a specific test set
def evaluate_model(clf, X_test, y_test, file_name):
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy for {file_name}: {accuracy}")
    
    # Print label distributions
    print(f"Test label distribution: {Counter(y_test)}")
    print(f"Prediction label distribution: {Counter(y_pred)}")
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    return accuracy

# Main function to train and evaluate on each CSV file
def main():
    csv_folder_path = r'C:\\Users\\Anugga\\Downloads\\\SUPINE FLEXION DATA\\supine flexion (5-6)\\*.csv'  # Update this path
    accuracies = []

    thresholds = calculate_thresholds(csv_folder_path)
    all_file_names = [file_name for file_name in os.listdir(csv_folder_path) if file_name.endswith(".csv")]

    for file_name in all_file_names:
        file_path = os.path.join(csv_folder_path, file_name)

        X_test, _ = load_data_from_csv(file_path)
        y_test = generate_labels(X_test, thresholds)

        # Prepare the training set
        X_train, y_train = [], []
        
        for other_file in all_file_names:
            if other_file != file_name:
                other_file_path = os.path.join(csv_folder_path, other_file)
                X_other, _ = load_data_from_csv(other_file_path)
                y_other = generate_labels(X_other, thresholds)

                X_train.extend(X_other)
                y_train.extend(y_other)

        # Print label distribution in the training set
        print(f"Training label distribution for {file_name}: {Counter(y_train)}")
        
        clf = train_model(X_train, y_train)
        accuracy = evaluate_model(clf, X_test, y_test, file_name)
        accuracies.append(accuracy)

    avg_accuracy = sum(accuracies) / len(accuracies)
    print(f"\nAverage accuracy across all CSVs: {avg_accuracy}")

# Run the main function
if __name__ == "__main__":
    main()