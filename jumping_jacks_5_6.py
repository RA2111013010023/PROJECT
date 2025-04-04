import sys
import pandas as pd
import numpy as np
import mediapipe as mp
import cv2
import joblib
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
import glob  # Added to handle wildcard paths

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Function to process video and save features to CSV
def process_video(video_path, output_csv_path):
    cap = cv2.VideoCapture(video_path)
    jumps, accuracy, time_diff_arms, time_diff_legs = [], [], [], []

    is_jumping = False
    jump_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

            left_arm_angle = calculate_angle(left_shoulder, left_hip, left_knee)
            right_arm_angle = calculate_angle(right_shoulder, right_hip, right_knee)
            left_leg_angle = calculate_angle(left_hip, left_knee, [left_knee[0], left_knee[1] + 0.1])
            right_leg_angle = calculate_angle(right_hip, right_knee, [right_knee[0], right_knee[1] + 0.1])

            if left_arm_angle < 160 and right_arm_angle < 160 and not is_jumping:
                is_jumping = True
            elif left_arm_angle > 160 and right_arm_angle > 160 and is_jumping:
                is_jumping = False
                jump_count += 1

            jumps.append(jump_count)
            accuracy.append((left_arm_angle + right_arm_angle) / 2)
            time_diff_arms.append(abs(left_arm_angle - right_arm_angle))
            time_diff_legs.append(abs(left_leg_angle - right_leg_angle))

    cap.release()

    df = pd.DataFrame({
        'jumps': jumps,
        'accuracy': accuracy,
        'time_diff_arms': time_diff_arms,
        'time_diff_legs': time_diff_legs
    })

    df.to_csv(output_csv_path, index=False)

def generate_labels(df, mean_accuracy, std_accuracy, mean_time_diff_arms, std_time_diff_arms, mean_time_diff_legs, std_time_diff_legs):
    labels = []
    for _, row in df.iterrows():
        accuracy = row['accuracy']
        time_diff_arms = row['time_diff_arms']
        time_diff_legs = row['time_diff_legs']
        
        if accuracy < mean_accuracy - std_accuracy:
            labels.append(0)
        else:
            if time_diff_arms <= mean_time_diff_arms + std_time_diff_arms and time_diff_legs <= mean_time_diff_legs + std_time_diff_legs:
                labels.append(2)
            else:
                labels.append(1)
    return labels

def load_data():
    # Gather all CSV files in the specified directory
    csv_files = glob.glob(r'C:\\Users\\Anugga\\Downloads\\JUMPING JACKS DATAS\\Jumping jacks (5-6)\\*.csv')
    # Concatenate all CSV files into a single DataFrame
    data = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)
    return data

def train_exercise_classifier():
    data = load_data()
    data['label'] = 1
    X = data[['jumps', 'time_diff_arms', 'time_diff_legs', 'accuracy']]
    y = data['label']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, 'exercise_classifier_5_6.pkl')
    print("Exercise classifier trained and saved as exercise_classifier_5_6.pkl")

def train_model():
    data = load_data()
    mean_accuracy = data['accuracy'].mean()
    std_accuracy = data['accuracy'].std()
    mean_time_diff_arms = data['time_diff_arms'].mean()
    std_time_diff_arms = data['time_diff_arms'].std()
    mean_time_diff_legs = data['time_diff_legs'].mean()
    std_time_diff_legs = data['time_diff_legs'].std()
    
    data['label'] = generate_labels(data, mean_accuracy, std_accuracy, mean_time_diff_arms, std_time_diff_arms, mean_time_diff_legs, std_time_diff_legs)
    X = data[['jumps', 'time_diff_arms', 'time_diff_legs', 'accuracy']]
    y = data['label']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, 'model_5_6.pkl')
    print("Model trained and saved as model_5_6.pkl")

def load_exercise_classifier():
    return joblib.load('exercise_classifier_5_6.pkl')

def load_model():
    return joblib.load('model_5_6.pkl')

def predict(file_path):
    exercise_model = load_exercise_classifier()
    input_df = pd.read_csv(file_path)
    input_X = input_df[['jumps', 'time_diff_arms', 'time_diff_legs', 'accuracy']]
    
    if input_X.empty:
        print("Error: No data to predict.")
        return input_df, 0, 0
    
    exercise_prediction = exercise_model.predict(input_X)
    if Counter(exercise_prediction).most_common(1)[0][0] == 0:
        input_df['predicted_accuracy'] = 0
        input_df['accuracy_score'] = 'Not a jumping jacks exercise'
        return input_df, 0, 0
    
    model = load_model()
    training_data = load_data()
    mean_accuracy = training_data['accuracy'].mean()
    std_accuracy = training_data['accuracy'].std()
    mean_time_diff_arms = training_data['time_diff_arms'].mean()
    std_time_diff_arms = training_data['time_diff_arms'].std()
    mean_time_diff_legs = training_data['time_diff_legs'].mean()
    std_time_diff_legs = training_data['time_diff_legs'].std()
    
    input_df['label'] = generate_labels(input_df, mean_accuracy, std_accuracy, mean_time_diff_arms, std_time_diff_arms, mean_time_diff_legs, std_time_diff_legs)
    input_X = input_df[['jumps', 'time_diff_arms', 'time_diff_legs', 'accuracy']]
    
    if input_X.empty:
        print("Error: No data to predict after labeling.")
        return input_df, 0, 0
    
    predictions = model.predict(input_X)

    input_df['predicted_accuracy'] = predictions
    input_df['accuracy_score'] = input_df['predicted_accuracy'].apply(lambda x: 'Not able to initiate jumps' if x == 0 else ('Moves arms or legs segmentally while jumping/ difficulty performing a proper jump' if x == 1 else 'Performs jumps correctly'))
    
    majority_score = Counter(predictions).most_common(1)[0][0]
    total_jumps = input_df['jumps'].iloc[-1]
    
    return input_df, majority_score, total_jumps

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        train_exercise_classifier()
        train_model()
    else:
        video_path = sys.argv[1]
        output_csv_path = 'output_5_6.csv'
        process_video(video_path, output_csv_path)
        result, majority_score, total_jumps = predict(output_csv_path)
        print(f"Final majority score: {majority_score}\nTotal number of jumps: {total_jumps}")