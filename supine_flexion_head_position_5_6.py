import sys
import os
import glob
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
import mediapipe as mp
import cv2
import joblib

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def process_video(video_path, output_csv_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    cap = cv2.VideoCapture(video_path)
    
    data = []

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
            nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,
                    landmarks[mp_pose.PoseLandmark.NOSE.value].y]

            left_knee_angle = calculate_angle(left_hip, left_knee, [left_knee[0], left_knee[1] + 0.1])
            right_knee_angle = calculate_angle(right_hip, right_knee, [right_knee[0], right_knee[1] + 0.1])
            head_midline = np.abs(nose[0] - ((left_shoulder[0] + right_shoulder[0]) / 2))
            chin_tuck = np.abs(nose[1] - ((left_shoulder[1] + right_shoulder[1]) / 2))
            left_shoulder_lifted = 1 if left_shoulder[1] < left_hip[1] else 0
            right_shoulder_lifted = 1 if right_shoulder[1] < right_hip[1] else 0
            left_ankle_lifted = 1 if left_knee[1] < left_hip[1] else 0
            right_ankle_lifted = 1 if right_knee[1] < right_hip[1] else 0
            is_supine_flexion = 1
            secs_diff_lifting = 0
            angle_head_position = calculate_angle(left_shoulder, nose, right_shoulder)
            distance_forehead_knee = np.sqrt((nose[0] - left_knee[0])**2 + (nose[1] - left_knee[1])**2)
            left_hip_knee_angle = left_knee_angle
            right_hip_knee_angle = right_knee_angle

            data.append([cap.get(cv2.CAP_PROP_POS_MSEC) / 1000, left_knee_angle, right_knee_angle, head_midline, chin_tuck,
                         left_shoulder_lifted, right_shoulder_lifted, left_ankle_lifted, right_ankle_lifted,
                         is_supine_flexion, secs_diff_lifting, angle_head_position, distance_forehead_knee,
                         left_hip_knee_angle, right_hip_knee_angle])

    cap.release()

    df = pd.DataFrame(data, columns=['Time', 'Left_Knee_Angle', 'Right_Knee_Angle', 'Head_Midline', 'Chin_Tuck',
                                     'Left_Shoulder_Lifted', 'Right_Shoulder_Lifted', 'Left_Ankle_Lifted',
                                     'Right_Ankle_Lifted', 'Is_Supine_Flexion', 'Secs_Diff_Lifting',
                                     'Angle_Head_Position', 'Distance_Forehead_Knee', 'Left_Hip_Knee_Angle',
                                     'Right_Hip_Knee_Angle'])
    df.to_csv(output_csv_path, index=False)

def load_data_from_csv(file_path):
    df = pd.read_csv(file_path)
    avg_angle_head_position = df['Angle_Head_Position'].mean()
    features = df[['Time', 'Left_Knee_Angle', 'Right_Knee_Angle', 'Head_Midline', 'Chin_Tuck',
                   'Left_Shoulder_Lifted', 'Right_Shoulder_Lifted', 'Left_Ankle_Lifted',
                   'Right_Ankle_Lifted', 'Is_Supine_Flexion', 'Secs_Diff_Lifting',
                   'Angle_Head_Position', 'Distance_Forehead_Knee', 'Left_Hip_Knee_Angle',
                   'Right_Hip_Knee_Angle']].values
    
    return features, avg_angle_head_position

def generate_labels(df, mean_angle_head_position, std_angle_head_position):
    labels = []
    for _, row in df.iterrows():
        _, _, _, Head_Midline, _, _, _, _, _, Is_Supine_Flexion, _, Angle_Head_Position, _, _, _ = row
        
        if not Is_Supine_Flexion or Head_Midline == 0:
            labels.append(0)
        else:
            if Angle_Head_Position >= mean_angle_head_position + std_angle_head_position:
                labels.append(2)
            else:
                labels.append(1)
    
    return labels

def train_model():
    csv_folder_path = r'C:\\Users\\Anugga\\Downloads\\SUPINE FLEXION DATA\\supine flexion (5-6)\\*.csv'
    all_file_names = [file_name for file_name in glob.glob(csv_folder_path)]
    
    all_data = pd.concat([pd.read_csv(file_name) for file_name in all_file_names], ignore_index=True)
    mean_angle_head_position = all_data['Angle_Head_Position'].mean()
    std_angle_head_position = all_data['Angle_Head_Position'].std()
    
    X_train, y_train = [], []
    for file_name in all_file_names:
        df = pd.read_csv(file_name)
        X, _ = load_data_from_csv(file_name)
        y = generate_labels(df, mean_angle_head_position, std_angle_head_position)
        X_train.extend(X)
        y_train.extend(y)
    
    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight='balanced', n_jobs=-1)
    clf.fit(np.array(X_train), np.array(y_train))
    joblib.dump(clf, 'supine_flexion_head_position_5_6_model.pkl')
    print("Model trained and saved as supine_flexion_head_position_5_6_model.pkl")

def load_model():
    return joblib.load('supine_flexion_head_position_5_6_model.pkl')

def evaluate_model(clf, X_test, y_test, file_name, avg_angle):
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    majority_label = Counter(y_pred).most_common(1)[0][0]
    result_message = (
        f"\nAccuracy for {file_name}: {accuracy}\n"
        f"Majority predicted label for {file_name}: {majority_label}\n"
        f"Average Angle Head Position for {file_name}: {avg_angle:.2f}\n"
        f"Test label distribution: {Counter(y_test)}\n"
        f"Prediction label distribution: {Counter(y_pred)}\n"
        "Classification Report:\n"
        f"{classification_report(y_test, y_pred, zero_division=0)}"
    )
    return accuracy, result_message

def process_supine_flexion_head_position_5_6(input_path):
    if os.path.isfile(input_path) and input_path.endswith('.mp4'):
        # Process the video file
        output_csv_path = 'output_5_6.csv'
        process_video(input_path, output_csv_path)
        input_path = output_csv_path

    accuracies = []
    result_messages = []
    
    all_file_names = [file_name for file_name in glob.glob(os.path.join(input_path, "*.csv"))] if os.path.isdir(input_path) else [input_path]
    
    all_data = pd.concat([pd.read_csv(file_name) for file_name in all_file_names], ignore_index=True)
    mean_angle_head_position = all_data['Angle_Head_Position'].mean()
    std_angle_head_position = all_data['Angle_Head_Position'].std()
    
    for file_name in all_file_names:
        df = pd.read_csv(file_name)
        X_test, avg_angle = load_data_from_csv(file_name)
        y_test = generate_labels(df, mean_angle_head_position, std_angle_head_position)
        
        X_train, y_train = [], []
        for other_file in all_file_names:
            if other_file != file_name:
                df_other = pd.read_csv(other_file)
                X_other, _ = load_data_from_csv(other_file)
                y_other = generate_labels(df_other, mean_angle_head_position, std_angle_head_position)
                X_train.extend(X_other)
                y_train.extend(y_other)
        
        if os.path.exists('supine_flexion_head_position_5_6_model.pkl'):
            clf = load_model()
        else:
            clf = train_model()
        
        accuracy, result_message = evaluate_model(clf, np.array(X_test), np.array(y_test), file_name, avg_angle)
        accuracies.append(accuracy)
        result_messages.append(result_message)
    
    avg_accuracy = sum(accuracies) / len(accuracies)
    final_result_message = f"\nAverage accuracy across all CSVs: {avg_accuracy}\n" + "\n".join(result_messages)
    
    return final_result_message

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        train_model()
    else:
        input_path = sys.argv[1]
        result_message = process_supine_flexion_head_position_5_6(input_path)
        print(result_message)