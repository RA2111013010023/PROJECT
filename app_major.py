from flask import Flask, render_template, request, redirect, url_for, flash
import os
import subprocess

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Define the available exercises and subcategories
exercises = {
    'Supine flexion': ['Simultaneous Lifting', 'Head Position', 'Upper Extremity', 'Lower Extremity'],
    'One Leg Standing': ['Left Leg and Eyes Opened', 'Left Leg and Eyes Closed', 'Right Leg and Eyes Opened', 'Right Leg and Eyes Closed'],
    'Jumping Feet Together': [],
    'Jumping Jacks': [],
    'Prone Extension': ['Simultaneous lifting', 'Head position', 'Upper extremity', 'Lower extremity', 'Knee position'],
    'Standing on Toes': ['Front view eyes closed', 'Front view eyes open', 'Side view eyes closed', 'Side view eyes open']
}

# Define the available age categories
age_categories = ['Category 1: Age 5 and 6', 'Category 2: Age 7 and 8', 'Category 3: Age 9 and 10', 'Category 4: Age 11 and 12']

@app.route('/')
def index():
    return render_template('index.html', exercises=exercises, age_categories=age_categories)

@app.route('/upload', methods=['POST'])
def upload():
    exercise = request.form.get('exercise')
    subcategory = request.form.get('subcategory')
    age_category = request.form.get('age_category')
    video = request.files['video']

    if video:
        video_path = os.path.join('uploads', video.filename)
        video.save(video_path)

        # Call the appropriate backend based on the selected exercise, subcategory, and age category
        backend_script = None

        if exercise in exercises:
            if subcategory in exercises[exercise]:
                if age_category == 'Category 1: Age 5 and 6':
                    backend_script = os.path.join('backends', exercise.replace(" ", "_").lower(), subcategory.replace(" ", "_").lower(), f'{exercise.replace(" ", "_").lower()}_{subcategory.replace(" ", "_").lower()}_5_6.py')
                elif age_category == 'Category 2: Age 7 and 8':
                    backend_script = os.path.join('backends', exercise.replace(" ", "_").lower(), subcategory.replace(" ", "_").lower(), f'{exercise.replace(" ", "_").lower()}_{subcategory.replace(" ", "_").lower()}_7_8.py')
                elif age_category == 'Category 3: Age 9 and 10':
                    backend_script = os.path.join('backends', exercise.replace(" ", "_").lower(), subcategory.replace(" ", "_").lower(), f'{exercise.replace(" ", "_").lower()}_{subcategory.replace(" ", "_").lower()}_9_10.py')
                elif age_category == 'Category 4: Age 11 and 12':
                    backend_script = os.path.join('backends', exercise.replace(" ", "_").lower(), subcategory.replace(" ", "_").lower(), f'{exercise.replace(" ", "_").lower()}_{subcategory.replace(" ", "_").lower()}_11_12.py')
            else:
                if age_category == 'Category 1: Age 5 and 6':
                    backend_script = os.path.join('backends', exercise.replace(" ", "_").lower(), f'{exercise.replace(" ", "_").lower()}_5_6.py')
                elif age_category == 'Category 2: Age 7 and 8':
                    backend_script = os.path.join('backends', exercise.replace(" ", "_").lower(), f'{exercise.replace(" ", "_").lower()}_7_8.py')
                elif age_category == 'Category 3: Age 9 and 10':
                    backend_script = os.path.join('backends', exercise.replace(" ", "_").lower(), f'{exercise.replace(" ", "_").lower()}_9_10.py')
                elif age_category == 'Category 4: Age 11 and 12':
                    backend_script = os.path.join('backends', exercise.replace(" ", "_").lower(), f'{exercise.replace(" ", "_").lower()}_11_12.py')

        if backend_script:
            result = subprocess.run(['python', backend_script, video_path], capture_output=True, text=True)
            result_message = result.stdout if result.returncode == 0 else result.stderr
            flash(result_message)
            return redirect(url_for('index'))

        # Placeholder for other backend scripts based on exercise and age category
        flash(f"Processed {exercise} - {subcategory} for {age_category}")
        return redirect(url_for('index'))

    flash('No video uploaded. Please try again.')
    return redirect(url_for('index'))

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)