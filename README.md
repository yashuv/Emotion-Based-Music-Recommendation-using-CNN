# Emotion-Based-Music-Recommendation-using-CNN
This project aims to develop an Emotion-Based Music Recommendation System using facial expressions, leveraging the power of CNNs to analyze facial images, classify the user's emotional state, and recommend music based on those emotions.

## Dataset
- Source of the dataset
  - The CNN model for the Emotion-Based Music Recommendation
System is developed using the Kaggle dataset called <a href="https://www.kaggle.com/datasets/msambare/fer2013">FER-2013</a>,
one of the most popular datasets for facial expression recognition
research. The dataset was created by collecting the results of a
Google image search for various emotions. The dataset provides a
valuable resource for training emotion recognition models.

- Statistics about the dataset
  - The **FER2013** dataset is divided into two parts: a training
dataset and a testing dataset. The training dataset contains
24,176 images while the testing dataset consists of 7,178
images. Each image in the dataset is grayscale and has a
resolution of 48x48 pixels. It aims to capture a wide range of
facial expressions representing seven different classes of
emotions: _Angry, Disgust, Fear, Happy, Sad, Surprise, and
Neutral_.

- Sample visualization of data points
   - <img src="https://github.com/yashuv/Emotion-Based-Music-Recommendation-using-CNN/assets/66567559/2c68ce08-db57-4729-994f-8ba2b6788351" alt="sample visualization of data points" class="center" />

## Project Methodology
**Input Image -> Pre-processing -> Face Detection -> Facial Feature Extraction -> Emotion Classification -> Music Recommendation**

1. Input Image: This is the first step where the user provides an image of their face as the input for the system. The image can be captured by a camera or uploaded from a file. The image should be clear and well-lit, and the face should be visible and not occluded by any objects or accessories.

2. Pre-processing: The input image is then transformed into a suitable format for the subsequent steps. The pre-processing may include resizing, cropping, rotating, converting to grayscale, histogram equalization, noise reduction, and other operations that enhance the quality and consistency of the image.

3. Face Detection: This is the third step where the system locates and extracts the face region from the pre-processed image. The face detection is done using a haar cascade algorithm for object detection in OpenCV in python.

4. Facial Feature Extraction: In this step, the system extracts and measures the facial features that are relevant for emotion recognition. The facial features may include the shape and position of the eyes, eyebrows, nose, mouth, chin, etc. The feature extraction can be done using various methods, such as Active Shape Models (ASM), Active Appearance Models (AAM), Local Binary Patterns (LBP), Gabor filters, etc. The output of this step is a vector or a matrix that represents the facial features in a numerical form.

5. Emotion Classification: This is the fifth step where the system classifies the facial emotion based on the extracted features. The emotion classification can be done using various methods, such as Support Vector Machines (SVM), K-Nearest Neighbors (KNN), Artificial Neural Networks (ANN), etc. The output of this step is a label or a score that indicates the emotion category or the intensity of the emotion. The emotion categories may include _Angry, Disgust, Fear, Happy, Sad, Surprise, and
Neutral_.

6. Music Recommendation: This is the final step where the system recommends music tracks that match the user’s facial emotion.

## Project Outcome
In this project application, the user provides their face to the model through a webcam, which detects the emotion and provides personalized music recommendations as output.
