from __future__ import print_function
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import time
import cv2
import numpy as np
import random
import dlib
import sys
from scipy.spatial import distance
from imutils import face_utils
import pygame #For playing sound
import tkinter as tk



from eye_key_funcs import *

# # ------------------------------------ Inputs
camera_ID = 1 # select webcam

width, height = 800, 500 # [pixels]
offset = (100, 80) # pixel offset (x, y) of keyboard coordinates

pygame.mixer.init()
pygame.mixer.music.load('audio/alert.wav')

true_labels = [1, 0, 1, 0, 1, 1, 0, 0, 1, 1]
predicted_labels = [1, 0, 1, 0, 1, 1, 0, 1, 1, 0]

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])

    ear = (A+B) / (2*C)
    return ear

pygame.mixer.init()
pygame.mixer.music.load('audio/alert.wav')

drowsiness_occurrences = []

#Minimum threshold of eye aspect ratio below which alarm is triggerd
EYE_ASPECT_RATIO_THRESHOLD = 0.3

#Minimum consecutive frames for which eye ratio is below threshold for alarm to be triggered
EYE_ASPECT_RATIO_CONSEC_FRAMES = 40

#COunts no. of consecutuve frames below threshold value
COUNTER = 0

resize_eye_frame = 4.5 # scaling factor for window's size
resize_frame = 0.3 # scaling factor for window's size

# # ------------------------------------

# ------------------------------------------------------------------- INITIALIZATION
# Initialize the camera
camera = init_camera(camera_ID = camera_ID)
cockpit = cv2.imread('cockpit.jpg')
cockpit1 = cv2.imread('cockpit1.jpg')

# take size screen

size_screen = (camera.get(cv2.CAP_PROP_FRAME_HEIGHT), camera.get(cv2.CAP_PROP_FRAME_WIDTH))

# make a page (2D frame) to write & project
#keyboard_page = make_black_page(size = size_screen)
#calibration_page = make_black_page(size = size_screen)

# Initialize keyboard
#key_points = get_keyboard(width  = width ,
#                       height = height ,
#                       offset = offset )

# upload face/eyes predictors
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# -------------------------------------------------------------------
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")

# ------------------------------------------------------------------- CALIBRATION
corners = [(offset),
           (width+offset[0], height + offset[1]),
           (width+offset[0], offset[1]),
           (offset[0], height + offset[1])]
calibration_cut = []
corner  = 0

while(corner<4): # calibration of 4 corners

    ret, frame = camera.read()   # Capture frame
    frame = adjust_frame(frame)  # rotate / flip

    gray_scale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

    # messages for calibration
    cv2.putText(cockpit, 'calibration: look at the circle and blink', tuple((np.array(size_screen)/7).astype('int')), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), 1)
    cv2.circle(cockpit, corners[corner], 20, (0, 255, 0), -1)

    # detect faces in frame
    faces = detector(gray_scale_frame)
    if len(faces)> 1:
        print('please avoid multiple faces.')
        sys.exit()

    for face in faces:
        display_box_around_face(frame, [face.left(), face.top(), face.right(), face.bottom()], 'green', (20, 40))

        landmarks = predictor(gray_scale_frame, face) # find points in face
        display_face_points(frame, landmarks, [0, 68], color='white') # draw face points

        # get position of right eye and display lines
        right_eye_coordinates= get_eye_coordinates(landmarks, [42, 43, 44, 45, 46, 47])
        display_eye_lines(frame, right_eye_coordinates, 'green')

    # define the coordinates of the pupil from the centroid of the right eye
        pupil_coordinates = np.mean([right_eye_coordinates[2], right_eye_coordinates[3]], axis = 0).astype('int')

        if is_blinking(right_eye_coordinates):
            calibration_cut.append(pupil_coordinates)

        # visualize message
            cv2.putText(cockpit, 'ok',
                        tuple(np.array(corners[corner])-5), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 255), 2)
            # to avoid is_blinking=True in the next frame
            time.sleep(0.3)
            corner = corner + 1

    print(calibration_cut, '    len: ', len(calibration_cut))
    show_window('projection', cockpit)
    show_window('frame', cv2.resize(frame,  (640, 360)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
# -------------------------------------------------------------------

# ------------------------------------------------------------------- PROCESS CALIBRATION
# find limits & offsets for the calibrated frame-cut
x_cut_min, x_cut_max, y_cut_min, y_cut_max = find_cut_limits(calibration_cut)
offset_calibrated_cut = [ x_cut_min, y_cut_min ]

# ----------------- message for user
print('------------------Good Going----------------------')
cv2.putText(cockpit, 'calibration done.',
            tuple((np.array(size_screen)/5).astype('int')), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 255), 2)
show_window('projection', cockpit)
cv2.waitKey(0)
cv2.destroyAllWindows()

print('-------------cockpit appearing------------------------')
# -------------------------------------------------------------------

# ------------------------------------------------------------------- WRITING
pressed_key = True
# key_on_screen = " "
#string_to_write = "text: "
while(True):

    ret, frame = camera.read()   # Capture frame
    frame = adjust_frame(frame)  # rotate / flip

    cut_frame = np.copy(frame[y_cut_min:y_cut_max, x_cut_min:x_cut_max, :])

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detect facial points through detector function
    faces = detector(gray, 0)

    #Detect faces through haarcascade_frontalface_default.xml
    face_rectangle = face_cascade.detectMultiScale(gray, 1.3, 5)

    # make & display on frame the keyboard
    #cockpit = cv2.imshow('cockpit.jpg')
    #dysplay_keyboard(img = keyboard_page, keys = key_points)
    #text_page = make_white_page(size = (200, 800))
    true_label = 1 if is_blinking(right_eye_coordinates) else 0

    # Assuming you have a predicted label (e.g., based on your system's logic)
    predicted_label = 1 if is_blinking(right_eye_coordinates) else 0

    # Append to the lists
    true_labels.append(true_label)
    predicted_labels.append(predicted_label)

    gray_scale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # gray-scale to work with

    faces = detector(gray_scale_frame)  # detect faces in frame
    if len(faces)> 1:
        print('please avoid multiple faces..')
        sys.exit()

    for face in faces:
        
        display_box_around_face(frame, [face.left(), face.top(), face.right(), face.bottom()], 'green', (20, 40))

        

        landmarks = predictor(gray_scale_frame, face) # find points in face
        display_face_points(frame, landmarks, [0, 68], color='red') # draw face points

        # get position of right eye and display lines
        right_eye_coordinates = get_eye_coordinates(landmarks, [42, 43, 44, 45, 46, 47])
        display_eye_lines(frame, right_eye_coordinates, 'green')


        #Drowsiness detection
        
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        #Get array of coordinates of leftEye and rightEye
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        #Calculate aspect ratio of both eyes
        leftEyeAspectRatio = eye_aspect_ratio(leftEye)
        rightEyeAspectRatio = eye_aspect_ratio(rightEye)

        eyeAspectRatio = (leftEyeAspectRatio + rightEyeAspectRatio) / 2

        #Use hull to remove convex contour discrepencies and draw eye shape around eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (255, 255, 255), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (255, 255, 255), 1)

        #Detect if eye aspect ratio is less than threshold
        if(eyeAspectRatio < EYE_ASPECT_RATIO_THRESHOLD):
            COUNTER += 1
            #If no. of frames is greater than threshold frames,
            if COUNTER >= EYE_ASPECT_RATIO_CONSEC_FRAMES:
                pygame.mixer.music.play(-1)
                cv2.putText(cockpit, "You are Drowsy", (150,200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2)
        else:
            pygame.mixer.music.stop()
            COUNTER = 0
    



    # define the coordinates of the pupil from the centroid of the right eye
    pupil_on_frame = np.mean([right_eye_coordinates[2], right_eye_coordinates[3]], axis = 0).astype('int')

    # work on the calbrated cut-frame
    pupil_on_cut = np.array([pupil_on_frame[0] - offset_calibrated_cut[0], pupil_on_frame[1] - offset_calibrated_cut[1]])
    cv2.circle(cut_frame, (pupil_on_cut[0], pupil_on_cut[1]), int(take_radius_eye(right_eye_coordinates)/1.5), (255, 0, 0), 3)

    if pupil_on_cut_valid(pupil_on_cut, cut_frame):

        pupil_on_cockpit= project_on_page(img_from = cut_frame[:,:, 0], # needs a 2D image for the 2D shape
                                            img_to = cockpit[:,:, 0], # needs a 2D image for the 2D shape
                                            point = pupil_on_cut)

        # draw circle at pupil_on_keyboard on the keyboard
        cv2.circle(cockpit, (pupil_on_cockpit[0], pupil_on_cockpit[1]), 20, (0, 255, 0), 2)

        '''if is_blinking(right_eye_coordinates):

            pressed_key = identify_key(key_points = key_points, coordinate_X = pupil_on_keyboard[1], coordinate_Y = pupil_on_keyboard[0])

            if pressed_key:
                if pressed_key=='del':
                    string_to_write = string_to_write[: -1]
                else:
                    string_to_write = string_to_write + pressed_key

            time.sleep(0.3) # to avoid is_blinking=True in the next frame

    # print on screen the string
    cv2.putText(text_page, string_to_write,
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2,(0, 0, 0), 5)'''

    # visualize windows
 
    show_window('cockpit', cockpit1)
    show_window('projection', cockpit)
    show_window('frame', cv2.resize(frame, (int(frame.shape[1] *resize_frame), int(frame.shape[0] *resize_frame))))
    show_window('cut_frame', cv2.resize(cut_frame, (int(cut_frame.shape[1] *resize_eye_frame), int(cut_frame.shape[0] *resize_eye_frame))))
    #save the image of the cockpit
    cv2.imwrite("cockpit.png", cockpit)
    #plt.savefig("tracked_report.png", cockpit)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Calculate metrics
#num_samples = len(true_labels)
accuracy = accuracy_score(true_labels, predicted_labels) * 100
precision = precision_score(true_labels, predicted_labels) * 100
recall = recall_score(true_labels, predicted_labels) * 100
f1 = f1_score(true_labels, predicted_labels) * 100

# Confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

# ROC curve
fpr, tpr, _ = roc_curve(true_labels, predicted_labels)
roc_auc = auc(fpr, tpr)

# Pie chart for drowsiness occurrences
drowsy_percentage = 30
non_drowsy_percentage = 100 - drowsy_percentage

labels = 'Drowsy', 'Non-Drowsy'
sizes = [drowsy_percentage, non_drowsy_percentage]
colors = ['#ff9999', '#66b3ff']
explode = (0.1, 0)  # explode 1st slice

# Tracking cockpit image
cockpit_image = cv2.imread('cockpit.jpg')

# Organizing plots and information on a single image
plt.figure(figsize=(10, 8))

# Confusion Matrix
plt.subplot(2, 2, 1)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")

# ROC Curve
plt.subplot(2, 2, 2)
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")

# Pie chart for drowsiness occurrences
plt.subplot(2, 2, 3)
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Drowsiness Occurrences')

# Tracking cockpit image
plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(cockpit, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Tracking Cockpit Image')

# Text information
print (f"Accuracy: {accuracy:.2f}%")
print (f"Precision: {precision:.2f}%")
print (f"Recall: {recall:.2f}%")
print (f"F1 Score: {f1:.2f}%")

plt.text(0.5, 0.02, f"Accuracy: {accuracy:.2f}%\nPrecision: {precision:.2f}%\nRecall: {recall:.2f}%\nF1 Score: {f1:.2f}%", ha='center', va='center', fontsize=12, color='white')

plt.tight_layout()

plt.savefig('result_image.png', dpi=300)
plt.show()
#-------------------------------------------

shut_off(camera) # Shut camera / windows off
