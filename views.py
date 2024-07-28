# # # smapp/views.py
# # import cv2
# # from django.shortcuts import render
# # from django.http import JsonResponse
# # import os
# # import numpy as np
# # import tensorflow as tf
# # import pickle
# # from .utils import predict_cropped_image, preprocess_cropped_image  # Import functions from utils
# #
# # def load_model_and_label_encoder():
# #     model_path = os.path.join(r'C:\Users\91859\PycharmProjects\Attendance\Smart_attendance\smproject\smapp\static\model_files\fine_tuned_model.h5')
# #     label_encoder_path = os.path.join(r'C:\Users\91859\PycharmProjects\Attendance\Smart_attendance\smproject\smapp\static\model_files\label_encoder.pkl')
# #
# #     model = tf.keras.models.load_model(model_path)
# #
# #     with open(label_encoder_path, 'rb') as file:
# #         label_encoder = pickle.load(file)
# #
# #     return model, label_encoder
# #
# # def your_view(request):
# #     model, label_encoder = load_model_and_label_encoder()
# #
# #     if request.method == 'POST' and 'image' in request.FILES:
# #         uploaded_image = request.FILES['image']
# #         # Use a temporary path to save the uploaded image
# #         image_path = os.path.join('media', 'uploads', uploaded_image.name)
# #
# #         # Save the uploaded image to a file
# #         with open(image_path, 'wb') as f:
# #             for chunk in uploaded_image.chunks():
# #                 f.write(chunk)
# #
# #         # Predict the label of the image
# #         predicted_label, _ = predict_cropped_image(image_path, model, label_encoder)
# #
# #         return JsonResponse({'predicted_label': predicted_label})
# #
# #     return render(request, 'home.html')
# #
# #
# # def process_video_frame(frame):
# #     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# #
# #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# #     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
# #
# #     for (x, y, w, h) in faces:
# #         face = frame[y:y + h, x:x + w]
# #         face = preprocess_cropped_image(face)
# #         face = np.expand_dims(face, axis=0)
# #         preds = model.predict(face)
# #         predicted_label_index = np.argmax(preds)
# #         predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]
# #
# #         # Draw rectangle and label on the frame
# #         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
# #         cv2.putText(frame, predicted_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
# #
# #         # Mark attendance
# #         mark_attendance(predicted_label)
# #
# #     return frame
# #
# #
# # def mark_attendance(predicted_label):
# #     import sqlite3
# #     conn = sqlite3.connect('db.sqlite3')  # Adjust path as needed
# #     cursor = conn.cursor()
# #
# #     cursor.execute("INSERT INTO attendance (student_name, date) VALUES (?, date('now'))", (predicted_label,))
# #
# #     conn.commit()
# #     conn.close()
# #
# #
# # def video_feed(request):
# #     # Open a connection to the camera
# #     cap = cv2.VideoCapture(0)
# #     while True:
# #         ret, frame = cap.read()
# #         if not ret:
# #             break
# #
# #         frame = process_video_frame(frame)
# #         cv2.imshow('Video', frame)
# #
# #         if cv2.waitKey(1) & 0xFF == ord('q'):
# #             break
# #
# #     cap.release()
# #     cv2.destroyAllWindows()
# #
# #     return render(request, 'video_feed.html')
# # import pickle
# #
# # import cv2
# # import numpy as np
# # import tensorflow as tf
# # from django.shortcuts import render
# # from django.http import JsonResponse
# # from .utils import preprocess_cropped_image, predict_cropped_image
# # import os
# #
# #
# # def load_model_and_label_encoder():
# #     model_path = os.path.join(r'C:\Users\91859\PycharmProjects\Attendance\Smart_attendance\smproject\smapp\static\model_files\fine_tuned_model.h5')
# #     label_encoder_path = os.path.join(r'C:\Users\91859\PycharmProjects\Attendance\Smart_attendance\smproject\smapp\static\model_files\label_encoder.pkl')
# #
# #     model = tf.keras.models.load_model(model_path)
# #
# #     with open(label_encoder_path, 'rb') as file:
# #         label_encoder = pickle.load(file)
# #
# #     return model, label_encoder
# #
# #
# # model, label_encoder = load_model_and_label_encoder()
# #
# #
# # def process_video_frame(frame):
# #     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# #
# #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# #     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
# #
# #     for (x, y, w, h) in faces:
# #         face = frame[y:y + h, x:x + w]
# #         face = preprocess_cropped_image(face)
# #         face = np.expand_dims(face, axis=0)
# #
# #         preds = model.predict(face)
# #         predicted_label_index = np.argmax(preds)
# #         predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]
# #
# #         # Draw rectangle and label on the frame
# #         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
# #         cv2.putText(frame, predicted_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
# #
# #         # Mark attendance
# #         mark_attendance(predicted_label)
# #
# #     return frame
# #
# #
# # def mark_attendance(predicted_label):
# #     import sqlite3
# #     conn = sqlite3.connect('db.sqlite3')  # Adjust path as needed
# #     cursor = conn.cursor()
# #
# #     cursor.execute("INSERT INTO attendance (student_name, date) VALUES (?, date('now'))", (predicted_label,))
# #
# #     conn.commit()
# #     conn.close()
# #
# #
# # def video_feed(request):
# #     model, label_encoder = load_model_and_label_encoder()
# #
# #     if request.method == 'POST':
# #         # Assuming you are sending image data directly from the POST request
# #         image_file = request.FILES.get('image')  # This should be the file uploaded through the form
# #
# #         if image_file:
# #             # Convert the image data to a numpy array
# #             image = np.frombuffer(image_file.read(), np.uint8)
# #             image = cv2.imdecode(image, cv2.IMREAD_COLOR)
# #
# #             # Predict the label
# #             true_label, predicted_label, processed_image = predict_cropped_image(image, model, label_encoder)
# #
# #             return JsonResponse({'true_label': true_label, 'predicted_label': predicted_label})
# #
# #     return render(request, 'home.html')
# # from django.shortcuts import render
# #
# # def upload_image(request):
# #     return render(request, 'upload_form.html')
#
# import pickle
# import cv2
# import numpy as np
# import tensorflow as tf
# from django.shortcuts import render
# from django.http import StreamingHttpResponse
# from .utils import predict_cropped_image, preprocess_cropped_image
# import os
#
# def load_model_and_label_encoder():
#     model_path = os.path.join(r'C:\Users\91859\PycharmProjects\Attendance\Smart_attendance\smproject\smapp\static\model_files\fine_tuned_model.h5')
#     label_encoder_path = os.path.join(r'C:\Users\91859\PycharmProjects\Attendance\Smart_attendance\smproject\smapp\static\model_files\label_encoder.pkl')
#
#     model = tf.keras.models.load_model(model_path)
#
#     with open(label_encoder_path, 'rb') as file:
#         label_encoder = pickle.load(file)
#
#     return model, label_encoder
#
# model, label_encoder = load_model_and_label_encoder()
#
#
def mark_attendance(predicted_label):
    import sqlite3
    conn = sqlite3.connect('C:\\Users\\91859\\PycharmProjects\\Attendance\\Smart_attendance\\smproject\\smproject\\db.sqlite3')
  # Adjust path as needed
    cursor = conn.cursor()

    cursor.execute("INSERT INTO attendance (student_name, date) VALUES (?, date('now'))", (predicted_label,))

    conn.commit()
    conn.close()
#
#
# # def is_student_registered(predicted_label):
# #     import sqlite3
# #     conn = sqlite3.connect('C:\\Users\\91859\\PycharmProjects\\Attendance\\Smart_attendance\\smproject\\smproject\\db.sqlite3')
# #     cursor = conn.cursor()
# #
# #     cursor.execute("SELECT * FROM Attendance WHERE name = ?", (predicted_label,))
# #     student = cursor.fetchone()
# #     conn.close()
# #
# #     return student is not None

# def process_video_frame(frame):
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
#
#     for (x, y, w, h) in faces:
#         face = frame[y:y + h, x:x + w]
#         face = preprocess_cropped_image(face)
#         face = np.expand_dims(face, axis=0)
#
#         preds = model.predict(face)
#         predicted_label_index = np.argmax(preds)
#         predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]
#
#         # Check if the predicted label matches a student in the database
#         # if is_student_registered(predicted_label):
#         #     mark_attendance(predicted_label)
#         # else:
#         #     print(f"Unrecognized student: {predicted_label}")
#
#         # Draw rectangle and label on the frame
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#         cv2.putText(frame, predicted_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#
#     return frame
#
# def generate_frames():
#     cap = cv2.VideoCapture(0)  # Use the default camera
#
#     while True:
#         success, frame = cap.read()
#         if not success:
#             break
#
#         # Process the video frame
#         frame = process_video_frame(frame)
#
#         # Encode the frame in JPEG format
#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()
#
#         # Yield the frame to be displayed
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#
#     cap.release()
#
# def video_feed(request):
#     return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')
#
# def home(request):
#     return render(request, 'home.html')

import cv2
import numpy as np
import tensorflow as tf
from django.http import StreamingHttpResponse
from django.shortcuts import render
from .utils import recognize_faces, preprocess_cropped_image, load_model_and_label_encoder
from .models import Attendance, UnrecognizedFace

# Load the model and label encoder once when the server starts
model, label_encoder = load_model_and_label_encoder()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def video_feed(request):
    def generate_frames():
        # Open the camera
        camera = cv2.VideoCapture(0)  # Use 0 for the default camera

        while True:
            success, frame = camera.read()  # Read a frame from the camera
            if not success:
                break

            # Process the frame for face recognition
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face_image = frame[y:y + h, x:x + w]
                predicted_label, prob = recognize_faces(face_image, model, label_encoder)

                # Set a confidence threshold
                if prob > 0.25:
                    print(f"Recognized: {predicted_label} with probability {prob:.2f}")
                else:
                    predicted_label = "Unknown"

                # Draw rectangle around the face and put text
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f"{predicted_label} ({prob:.2f})", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            # Encode the frame in JPEG format
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the output frame in the format required for StreamingHttpResponse
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

# def home(request):
#     return render(request, 'home.html')  # Render your HTML template here
def index(request):
    return render(request, 'home.html')  # Render your HTML template here

def landing(request):
    return render(request, 'landing.html')

from django.shortcuts import render
from .models import Attendance, Student

def student_status(request):
    # Fetch the Student instance for the logged-in user
    try:
        student = Student.objects.get(name=request.user.username)  # Assuming username corresponds to Student's name
    except Student.DoesNotExist:
        student = None

    # If the student exists, fetch their attendance records
    if student:
        attendance_records = Attendance.objects.filter(student=student)
    else:
        attendance_records = []

    return render(request, 'student_status.html', {'attendance_records': attendance_records})


def admin_dashboard(request):
    # Fetch all attendance records
    attendance_records = Attendance.objects.all()
    return render(request, 'admin_dashboard.html', {'attendance_records': attendance_records})

def unrecognized_faces(request):
    # Fetch unrecognized faces for admin review
    unrecognized_faces = UnrecognizedFace.objects.all()
    return render(request, 'unrecognized_faces.html', {'unrecognized_faces': unrecognized_faces})

def resolve_unrecognized(request, face_id):
    # Logic to resolve unrecognized face (e.g., marking attendance manually)
    pass