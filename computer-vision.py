import cv2
import numpy as np

# Load pre-trained model for object detection
net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')

# Get the names of the output layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Load video
video = cv2.VideoCapture('sample_video.mp4')

# Create tracker
tracker = cv2.TrackerKCF_create()

# Initialize variables
frame_width = int(video.get(3))
frame_height = int(video.get(4))
output_video = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))
object_detected = False

while True:
    ret, frame = video.read()
    
    if not ret:
        break
    
    # Detect objects using YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    class_ids = []
    confidences = []
    boxes = []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5:
                # Object detected
                object_detected = True
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                
                # Calculate top-left and bottom-right coordinates of bounding box
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                
                boxes.append([x, y, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Track detected objects
    if object_detected:
        for box in boxes:
            x, y, width, height = box
            tracker.init(frame, (x, y, width, height))
            object_detected = False
    
    success, box = tracker.update(frame)
    
    if success:
        x, y, width, height = [int(i) for i in box]
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
    
    # Write frame to output video
    output_video.write(frame)
    
    cv2.imshow('Object Detection and Tracking', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
output_video.release()
cv2.destroyAllWindows()
