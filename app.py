from flask import Flask, render_template, Response
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time
import cv2
import mediapipe as mp
import numpy as np
import joblib
import tensorflow as tf
from twilio.rest import Client  # Twilio for SMS
from flask import request, jsonify

app = Flask(__name__)

# Load model and preprocessing tools
model = tf.keras.models.load_model('activity_classifier_optimized.h5')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Mediapipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# State variables
activity_start_time = None
last_activity = None
refresh_page = False

# Twilio credentials (replace with your values)
TWILIO_ACCOUNT_SID = 'ACd5561df9482d1f2afbb151637f7d7193'
TWILIO_AUTH_TOKEN = '413520f4ac256eb8ab2b77e1e2273085'
TWILIO_FROM_NUMBER = '+18653916106'  # Your Twilio number
TO_PHONE_NUMBER = '+919355961963'   # Recipient's phone number with country code


def preprocess_landmarks(landmarks):
    try:
        feature_vector = [
            [landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in landmarks
        ]
        feature_vector = np.array(feature_vector).flatten().reshape(1, -1)
        feature_vector = scaler.transform(feature_vector)
        return feature_vector
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None


def send_sms_alert(activity):
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            body=f"Alert: {activity} detected. Immediate attention required!",
            from_=TWILIO_FROM_NUMBER,
            to=TO_PHONE_NUMBER
        )
        print(f"SMS sent successfully for activity: {activity}, SID: {message.sid}")
    except Exception as e:
        print(f"Error sending SMS: {e}")


def send_email_alert(activity):
    global refresh_page
    try:
        sender_email = "riyakansal174@gmail.com"
        receiver_email = "japp.kaur28@gmail.com"
        password = "duge vxis stkk npfs"  # Use App Password

        subject = f"Abnormal Activity Detected: {activity}"
        body = f"Alert: {activity} detected. Immediate attention required!!!."

        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = receiver_email
        message["Subject"] = subject
        message.attach(MIMEText(body, "plain"))

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message.as_string())

        print(f"Email sent successfully for activity: {activity}")

        # Also send SMS
        send_sms_alert(activity)

        refresh_page = True

    except Exception as e:
        print(f"Error sending email or SMS: {e}")


def predict_activity(processed_landmarks):
    global activity_start_time, last_activity, refresh_page

    try:
        predictions = model.predict(processed_landmarks)
        predicted_class = np.argmax(predictions, axis=1)[0]
        activity = label_encoder.inverse_transform([predicted_class])[0]

        if activity in ["heart stroke", "headache", "falling", "Cough"]:
            if activity == last_activity:
                if time.time() - activity_start_time >= 10:
                    send_email_alert(activity)
                    print(f"Alert sent for {activity}.")
            else:
                last_activity = activity
                activity_start_time = time.time()
                print(f"Started tracking {activity}.")

        return activity
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Prediction Error"


# Setup webcam
cap = cv2.VideoCapture(0)
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

def gen():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No frame received from webcam. Exiting.")
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = results.pose_landmarks.landmark
            processed_landmarks = preprocess_landmarks(landmarks)
            activity = (
                predict_activity(processed_landmarks)
                if processed_landmarks is not None
                else "Processing Error"
            )
        else:
            activity = "No Person Detected"

        cv2.putText(image, f"Activity: {activity}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)
        ret, jpeg = cv2.imencode('.jpg', image)
        if not ret:
            continue
        frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/')
def index():
    global refresh_page
    if refresh_page:
        refresh_page = False
        return render_template('index.html', refresh=True)
    return render_template('index.html', refresh=False)


@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict', methods=['POST'])
def predict_from_flutter():
    try:
        data = request.json.get('keypoints')  # Expecting 132 keypoints
        if not data or len(data) != 132:
            return jsonify({'error': 'Invalid input length. Must be 132 values.'}), 400

        features = scaler.transform([data])
        prediction = model.predict(features)
        predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])[0]

        # Optional: Trigger alert logic here if needed
        if predicted_class in ["heart stroke", "headache", "falling", "Cough"]:
            print(f"ALERT (from Flutter): {predicted_class}")

        return jsonify({'activity': predicted_class})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, threaded=True) 

