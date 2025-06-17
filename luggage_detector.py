import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import sys

# Global variable for luggage classes to detect
luggage_classes = ['bag', 'suitcase', 'luggage', 'backpack', 'handbag', 'purse', 'case']

# Class name mapping for common variations
class_name_mapping = {
    'handbag': ['handbag', 'purse', 'bag'],
    'suitcase': ['suitcase', 'luggage', 'case'],
    'backpack': ['backpack', 'bag'],
    # Add more mappings as needed
}

def matches_luggage_class(class_name):
    """
    Check if a class name matches any of the luggage classes we're looking for
    """
    class_name = class_name.lower()
    
    # Direct match
    if class_name in luggage_classes:
        return True
        
    # Check mappings - if the model class name matches any of our mapped categories
    for model_class, variants in class_name_mapping.items():
        if class_name == model_class:
            return True
            
    # Check for partial matches
    for lc in luggage_classes:
        if lc in class_name:
            return True
            
    return False

def process_frame(frame, model, confidence_threshold=0.5):
    """
    Process a single frame to detect luggage items
    """
    # Use luggage classes from global variable
    global luggage_classes
    
    # Run YOLOv8 inference on the frame
    results = model(frame)
    
    # Process the results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get class name
            cls = int(box.cls[0])
            class_name = result.names[cls]
            
            # Only process if it's a luggage item
            if matches_luggage_class(class_name):
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Get confidence score
                conf = float(box.conf[0])
                
                # Only process if confidence is above threshold
                if conf > confidence_threshold:
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add label with confidence
                    label = f"{class_name}: {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Luggage Detection using YOLOv8')
    parser.add_argument('--source', type=str, default='0',
                      help='Video source (0 for webcam, or path to video file)')
    parser.add_argument('--confidence', type=float, default=0.5,
                      help='Confidence threshold for detection')
    parser.add_argument('--classes', type=str, default='bag,suitcase,luggage,backpack,handbag,purse,case',
                      help='Comma-separated list of luggage classes to detect')
    args = parser.parse_args()
    
    # Initialize the list of luggage classes from args
    global luggage_classes
    luggage_classes = [cls.strip().lower() for cls in args.classes.split(',')]
    print(f"Detecting luggage classes: {luggage_classes}")
    
    # Load YOLO model
    try:
        print("Trying to load YOLOv11 model...")
        model = YOLO('models\yolo11s.pt')  # Try to load YOLOv11 if available
        print("YOLOv11s model loaded successfully!")
    except Exception as e:
        print(f"Error loading YOLOv11 model: {e}")
        print("Falling back to YOLOv10 model...")

        # Fall back to YOLOv10 model
        try:
            model = YOLO('models\yolov10n.pt')
            print("YOLOv10 model loaded successfully!")
        except Exception as e:
            print(f"Error loading YOLOv10 model: {e}")
            print("Please run download_yolo_model.py to download YOLO models.")
            return
              # Print model class information
    class_names = model.model.names
    print("\nAvailable classes in the YOLO model:")
    luggage_classes_in_model = []
    for idx, name in class_names.items():
        if matches_luggage_class(name):
            print(f"  Class {idx}: {name} (✓ Will be detected)")
            luggage_classes_in_model.append(name.lower())
    
    # Warn about classes not in the model
    for cls in luggage_classes:
        if cls not in luggage_classes_in_model and not any(cls in name.lower() for name in class_names.values()):
            print(f"  Warning: '{cls}' not found in model classes")
    
    print(f"\nActually detecting luggage classes: {luggage_classes_in_model}")
    print("If your desired luggage items are not being detected, they may not be present in the model's classes.")
    
    # Initialize video capture
    if args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
    else:
        cap = cv2.VideoCapture(args.source)

    if not cap.isOpened():
        print("Error: Could not open video source")
        return

    print("Press 'q' or 'Esc' to quit")
    print("Press 'x' to close window")

    try:
        window_name = 'Luggage Detection'
        
        # Create a named window with specific properties that allow us to track close events
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Flag to track if window is closed
        running = True
        
        while running:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            processed_frame = process_frame(frame, model, args.confidence)

            # Display the frame
            cv2.imshow(window_name, processed_frame)
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            
            # Check if window was closed
            try:
                visible = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
                if visible < 1:  # Window was closed
                    print("Window closed by user")
                    break
            except:
                # Window doesn't exist anymore (closed by user)
                print("Window was closed")
                break
                
            # Check keyboard commands
            if key == ord('q') or key == 27:  # 'q' or 'Esc'
                break
            elif key == ord('x'):  # 'x' key
                break

    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        # Force close all windows
        for i in range(5):
            cv2.waitKey(1)
        sys.exit(0)

if __name__ == "__main__":
    main()