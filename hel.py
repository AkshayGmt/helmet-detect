import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
# Set a threshold for the minimum confidence level required for detection

CONFIDENCE = 0.5
IOU_THRESHOLD = 0.5
SCORE_THRESHOLD = 0.5
# Load YOLO object detection model and its configuration file

config_path = "yolov3-helmet.cfg"
weights = "yolov3-helmet.weights"
labels = open("helmet.names").read().strip().split("\n")

net = cv2.dnn.readNetFromDarknet(config_path, weights)
def model_output(path_name):
    image = cv2.imread(path_name)
    h,w = image.shape[:2]
        # Create a blob from the input frame and pass it through the network

    blob = cv2.dnn.blobFromImage(image, 1/255.0,(416,416), swapRB = True, crop = False)
    
    net.setInput(blob)  # Sets the new input value for the network
    
    ln = net.getLayerNames()
    print(ln)
    print('New value',net.getLayerNames())
    #ln is a list comprsisng all models in config file
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    # Run forward pass through YOLO model

    layer_outputs = net.forward(ln)
    boxes, confidences, class_ids = [], [], []
    for output in layer_outputs:
        for detection in output:
            # Get class ID and confidence
            scores = detection[5:]
            # Extract the class ID and confidence level of the current detection

            class_id = np.argmax(scores)
            #print(class_id)
            confidence = scores[class_id]
            # Filter out detections that don't meet the minimum confidence threshold

            if confidence>CONFIDENCE:
            # Extract the coordinates of the bounding box for the object

                box = detection[:4]*np.array([w,h,w,h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width/2))
                y = int(centerY - (height/2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    return boxes, confidences, class_ids

def detection_recognition(path_name):
    image = cv2.imread(path_name)
    boxes, confidences, class_ids = model_output(path_name)
    # Apply non-max suppression
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, IOU_THRESHOLD)
    font_scale = 1
    thickness= 1
    if len(idxs)>0:
        for i in idxs.flatten():
            x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
            cv2.rectangle(image, (x,y), (x+w, y+h), color = (255,20,147), thickness = thickness)
            object_img = image[y:y+h, x:x+w]
            
            # Display image with boxes
            cv2.imwrite( f'object1.jpg', object_img)

            text = f"{labels[class_ids[i]]}:{confidences[i]:.2f}"
            #Text  size
            (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale = font_scale, thickness = thickness)[0]
            text_offset_x = x
            text_offset_y = y - 5
            box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
            overlay = image.copy()
            #rectangle box create for object
            cv2.rectangle(overlay, box_coords[0], box_coords[1], color = (255,20,147), thickness = cv2.FILLED)
            image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
            #text apply in image
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale, color=(0, 0, 0), thickness=thickness)
            #image showing
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

detection_recognition("object.jpg")

