


import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt
import streamlit as st
import csv
import os
from datetime import datetime

# Load the pretrained YOLO model for people detection
people_model = YOLO("yolov8m.pt")  # YOLOv8 Medium for general object detection
ppe_model = YOLO("ppe_detector.pt")  # Your trained model for PPE detection

# Function to save compliance results to a CSV file with timestamp
def save_compliance_to_csv(results, filename="ppe_compliance_log.csv"):
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        
        # Write headers if file is new
        if not file_exists:
            writer.writerow(["Timestamp", "Person ID", "Compliance Status"])
        
        # Write detection results with timestamp
        for person_id, status in results.items():
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Get current timestamp
            writer.writerow([timestamp, person_id, status])
    
    st.success("Compliance log saved to CSV file.")

# Streamlit UI
st.title("PPE Compliance Detection")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

def detect_people_helmets(image):
    image = np.array(image)
    image_copy = image.copy()
    
    # Step 1: Detect people
    people_results = people_model(image)
    people_detections = people_results[0]
    
    person_boxes = []
    person_ids = {}
    person_count = 0
    
    for box in people_detections.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = people_model.names[int(box.cls)]
        conf = box.conf.item()
        
        if conf > 0.30 and label == "person":  # Confidence threshold
            person_count += 1
            person_id = f"Person {chr(64 + person_count)}"
            
            # Increase the height by a small percentage (e.g., 10%)
            y1 = max(0, y1 - int(0.2 * (y2 - y1)))

            person_boxes.append((x1, y1, x2, y2))
            person_ids[(x1, y1, x2, y2)] = person_id
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_copy, f"{person_id} ({conf:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Step 2: Detect helmets on detected people
    compliance_status = {}
    
    ppe_results = ppe_model(image)
    for box in ppe_results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = ppe_model.names[int(box.cls)]
        conf = box.conf.item()
        
        if conf > 0.50 and "helmet" in label:
            color = (255, 0, 0)
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image_copy, f"{label} ({conf:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            for person_box in person_boxes:
                px1, py1, px2, py2 = person_box
                if x1 >= px1 and y1 >= py1 and x2 <= px2 and y2 <= py2:
                    person_id = person_ids[person_box]
                    compliance_status[person_id] = "Complied"
    
    for person in person_ids.values():
        status = compliance_status.get(person, "Not Complied")
        compliance_status[person] = status
    
    return image_copy, person_count, compliance_status

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="PPE Compliance Image", use_column_width=True)
    
    processed_image, person_count, compliance_status = detect_people_helmets(image)
    
    st.image(processed_image, caption="Detection Results", use_column_width=True)
    st.write(f"Total People Detected: {person_count}")
    st.write("Helmet Compliance Summary:")
    for person, status in compliance_status.items():
        st.write(f"{person}: {status}")
    
    if st.button("Generate CSV Report"):
        save_compliance_to_csv(compliance_status)
