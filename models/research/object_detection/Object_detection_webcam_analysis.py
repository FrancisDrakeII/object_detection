
## Reference and Acknowledgement 
## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb


# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys


# Set up Twilio
from twilio.rest import Client

# SID and Token are stored as environmental variable in the system directory
account_sid = os.environ['TWILIO_ACCOUNT_SID']
auth_token = os.environ['TWILIO_AUTH_TOKEN']

client = Client(account_sid,auth_token)

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 4

## Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts 1, this corresponds to yellow, 2 correspond to white,
# 3 correspond to pod, 4 correspond to bud.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


# Define input and output tensors for the object detection classifier
# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes (numbers)
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Initialize webcam feed and define the resolution
video = cv2.VideoCapture(0)
ret = video.set(3,1280)
ret = video.set(4,720)

# Define counter for object detector (white, yellow ,pod and bud) and SMS message sending counter 
count1 = 0
count2 = 0
count3 = 0
count4 = 0
counter = 0

# I define this main function to perform the object detection/analysis.
# each frame will be captured.Then the function will do object detection 
# on each frame to draw results of the detection. Meanwhile, the 
def feature_detector(frame):
    
    # Initialize the counter globally
    global count1, count2, count3, count4, counter
    
    # Acquire frame and expand the frame
    frame_expanded = np.expand_dims(frame, axis=0)

     
    # Perform the object detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})


    # Draw the results of the detection 
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=3,
        min_score_thresh=0.60)

    # This loop will keep looking over the defined class label and count up if
    # the labels are detected. It also requires a minimum threshold (60% confidence
    # of the detection result for each label) in order for the counter to count up.
    for i in range (len(classes[0])):
        if (classes[0][i] ==1) and (scores[0][i] > 0.60):
            count1 += 1
        elif (classes[0][i] ==2) and (scores[0][i] > 0.60):
            count2 += 1
        elif (classes[0][i] ==3) and (scores[0][i] > 0.60):
            count3 += 1
        elif (classes[0][i] ==4) and (scores[0][i] > 0.60):
            count4 += 1

    # Display the result of the number of each detected label on the specified location
    # on each frame
    cv2.putText(frame,'white: ' + str(count1),(30,75),cv2.FONT_HERSHEY_COMPLEX,3,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(frame,'yellow: ' + str(count2),(30,145),cv2.FONT_HERSHEY_COMPLEX,3,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(frame,'pod: ' + str(count3),(30,215),cv2.FONT_HERSHEY_COMPLEX,3,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(frame,'bud: ' + str(count4),(30,285),cv2.FONT_HERSHEY_COMPLEX,3,(0,0,0),3,cv2.LINE_AA)
    
    counter+=1

    # This loop will send the result of the number of each detected label through the
    # Twilio API to the user. This service cost $0.004/message and I will keep looking
    # for more free API service such as smtlib and telegram in the future.
    # In this case, I set the counter be 50 to reduce the noise from the environment
    # to let the program know the analysis is done and the result can be forwarded to
    # me
    if counter > 50:
        message = client.messages.create(
            body = 'white ' + str(count1) + ' yellow ' + str(count2) + ' pod ' +str(count3) + ' bud ' + str(count4)
            ,
            from_='+16304488346',
            to='+19176575167'
            )
        counter = 0
        
    return frame
 
    
while(True):

    # Reset the counter for the next frame
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    
   
    # Load frame in real time into video feed
    ret, frame = video.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Initialize the function
    frame = feature_detector(frame)
    
    
    # Display all
    cv2.imshow('Object detector', frame)
    

    # Press 'q' to quit 
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
video.release()
cv2.destroyAllWindows()

