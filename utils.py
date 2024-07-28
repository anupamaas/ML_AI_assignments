import os
import pickle

import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import tensorflow as tf

def preprocess_cropped_image(face_image, target_size=(160, 160)):
    """
    Preprocesses the cropped face image for the model.

    Parameters:
        face_image (ndarray): The cropped face image.
        target_size (tuple): The target size for the model input.

    Returns:
        ndarray: The preprocessed face image.
    """
    # Resize the image
    face_image = cv2.resize(face_image, target_size)

    # Normalize the image
    face_image = face_image.astype('float32') / 255.0

    return face_image



def recognize_faces(face_image, model, label_encoder):
    """
    Recognizes faces in the provided face image using the given model and label encoder.

    Parameters:
        face_image (ndarray): The cropped face image to be recognized.
        model: The trained model for face recognition.
        label_encoder: The label encoder used to convert model outputs to class labels.

    Returns:
        tuple: A tuple containing the predicted label and the confidence probability.
    """
    # Preprocess the face image
    face_image = preprocess_cropped_image(face_image)  # Preprocess the image
    face_image = np.expand_dims(face_image, axis=0)  # Add batch dimension

    # Make predictions
    preds = model.predict(face_image)
    predicted_label_index = np.argmax(preds)
    predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]
    confidence = np.max(preds)

    return predicted_label, confidence

def preprocess_cropped_image(image, target_size=(160, 160)):
    # Ensure the image is in the correct format and resize it
    image = cv2.resize(image, target_size)
    image = image.astype('float') / 255.0
    image = img_to_array(image)
    return image

def predict_cropped_image(image, model, label_encoder, target_size=(160, 160)):
    # Placeholder for true label as this might not be used in real-time processing
    true_label = "Unknown"

    # Preprocess the image
    image = preprocess_cropped_image(image, target_size)
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Predict the label
    preds = model.predict(image)
    predicted_label_index = np.argmax(preds)
    predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]

    return true_label, predicted_label, image[0]
def load_model_and_label_encoder():
    model_path = os.path.join(r'C:\Users\91859\PycharmProjects\Attendance\Smart_attendance\smproject\smapp\static\model_files\fine_tuned_model.h5')
    label_encoder_path = os.path.join(r'C:\Users\91859\PycharmProjects\Attendance\Smart_attendance\smproject\smapp\static\model_files\label_encoder.pkl')

    model = tf.keras.models.load_model(model_path)

    with open(label_encoder_path, 'rb') as file:
        label_encoder = pickle.load(file)

    return model, label_encoder