import os
import requests
import argparse
from tqdm import tqdm
from ultralytics import YOLO

# Model URLs for YOLOv10-v11
base_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
YOLO_MODELS = {  
    # YOLOv10 models 
    "yolov10n.pt": base_url + "yolov10n.pt",
    "yolov10s.pt": base_url + "yolov10s.pt",
    "yolov10m.pt": base_url + "yolov10m.pt",
    
    # YOLOv11 models 
    "yolo11n.pt": base_url + "yolo11n.pt",
    "yolo11s.pt": base_url + "yolo11s.pt",
    "yolo11m.pt": base_url + "yolo11m.pt",
    "yolo11l.pt": base_url + "yolo11l.pt",
    "yolo11x.pt": base_url + "yolo11x.pt",
}

# List of luggage-related classes in COCO dataset
LUGGAGE_CLASSES = ['backpack', 'suitcase', 'handbag']

def download_file(url, destination):
    """
    Download a file from URL with progress bar
    """
    print(f"Downloading from {url} to {destination}")
    
    try:
        # Make the request with timeout
        response = requests.get(url, stream=True, timeout=30)
        
        # Check if the request was successful
        if response.status_code != 200:
            print(f"Error: Received status code {response.status_code}")
            return False
            
        # Get file size for progress bar
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        
        # Download with progress bar
        with open(destination, 'wb') as file, tqdm(
            desc=destination,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                bar.update(len(data))
                file.write(data)
        
        print(f"Successfully downloaded {os.path.basename(destination)}")
        return True
            
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False

def download_latest_yolo():
    """
    Download YOLOv10 and YOLOv11 models if they are missing
    """
    # Ensure models directory exists
    models_dir = ensure_models_directory()
    
    # Check which models are missing
    missing_models = []
    for model_name in YOLO_MODELS:
        model_path = os.path.join(models_dir, model_name)
        if not os.path.exists(model_path):
            missing_models.append(model_name)
    
    if not missing_models:
        print("All YOLOv10-v11 models are already downloaded.")
        return
    
    print(f"Missing models: {missing_models}")
    
    # Try to download missing models
    for model_name in missing_models:
        model_path = os.path.join(models_dir, model_name)
        print(f"Attempting to download {model_name}...")
        
        # Try via Ultralytics API first
        try:
            model = YOLO(model_name)
            source_path = model.ckpt_path
            print(f"Copying from {source_path} to {model_path}")
            import shutil
            shutil.copy2(source_path, model_path)
            print(f"{model_name}: Successfully downloaded")
        except Exception as e:
            print(f"API download failed: {str(e)[:50]}...")
            
            # Try direct download
            try:
                download_file(YOLO_MODELS[model_name], model_path)
            except Exception as e2:
                print(f"Direct download failed: {str(e2)[:50]}...")
      # Try to download models using Ultralytics API
    try:
        print("Attempting to download latest YOLO models via Ultralytics API...")
        
        # First try downloading a known model through Ultralytics API
        print("Downloading base YOLOv8n model as a starting point...")
        model = YOLO('yolov8n.pt')  # Start with a known model
        print(f"Successfully accessed base model at: {model.ckpt_path}")
        
        # Use this model's path to save to our models directory
        base_model_dest = os.path.join(models_dir, 'yolov8n.pt')
        if not os.path.exists(base_model_dest):
            print(f"Copying base model to models directory: {base_model_dest}")
            import shutil
            shutil.copy2(model.ckpt_path, base_model_dest)
        
        # Keep the classes display functionality
        display_model_classes(model)
        
        # Try to download newer model versions in priority order
        print("\nAttempting to download latest YOLO models...")
        
        # Try to download models in priority order
        for model_name in model_priority: # type: ignore
            # Skip yolov8n.pt as we already have it
            if model_name == 'yolov8n.pt':
                continue
                
            model_path = os.path.join(models_dir, model_name)
            if os.path.exists(model_path) and os.path.getsize(model_path) > 1000000:
                print(f" - {model_name}: Already exists at {model_path}")
                continue
                
            print(f" - Attempting to download {model_name}...")
            success = False
            
            # Try via API first
            try:
                temp_model = YOLO(model_name)
                # Copy the model to our models directory
                source_path = temp_model.ckpt_path
                print(f"   Copying from {source_path} to {model_path}")
                import shutil
                shutil.copy2(source_path, model_path)
                print(f" - {model_name}: Successfully downloaded and saved to {model_path}")
                success = True
                # Once we get a successful download, we can stop
                break
            except Exception as e:
                print(f" - {model_name}: Not available via API ({str(e)[:50]}...)")
                # Try direct download next
                try:
                    if model_name in YOLO_MODELS:
                        print(f"   Attempting direct download of {model_name}...")
                        if download_file(YOLO_MODELS[model_name], model_path):
                            print(f" - {model_name}: Successfully downloaded via direct link")
                            success = True
                            # Once we get a successful download, we can stop
                            break
                except Exception as e2:
                    print(f"   Direct download failed: {str(e2)[:50]}...")
        
        # Check if we successfully downloaded at least one model
        models_found = [f for f in os.listdir(models_dir) if f.endswith('.pt') and os.path.getsize(os.path.join(models_dir, f)) > 1000000]
        if models_found:
            print(f"\nAvailable models: {models_found}")
            return True
        
    except Exception as e:
        print(f"Error accessing Ultralytics models: {str(e)}")
        print("Falling back to manual download...")    # Define global models variable for use by other functions
    # Use the existing YOLO_MODELS dictionary that was defined at the top of the file
    global models
    models = YOLO_MODELS
    
    # Try direct download of a reliable model first
    reliable_models = ["yolov8n.pt", "yolov8s.pt"]
    for model_name in reliable_models:
        try:
            destination = os.path.join(models_dir, model_name)
            if not os.path.exists(destination):         
                print(f"\nAttempting direct download of reliable model {model_name}...")
                if model_name in YOLO_MODELS and download_file(YOLO_MODELS[model_name], destination):
                    print(f"Successfully downloaded {model_name}")
                    try:
                        model = YOLO(destination)
                        display_model_classes(model)
                        return True
                    except Exception as e:
                        print(f"Error loading downloaded model: {e}")
        except Exception as e:
            print(f"Error downloading {model_name}: {e}")
      # If automatic downloads failed, prompt user for manual choice
    print("\nAvailable models for direct download:")
    available_models = list(YOLO_MODELS.keys())
    available_models.sort()  # Sort alphabetically
    
    # Display in columns for better readability
    cols = 3
    rows = (len(available_models) + cols - 1) // cols
    for row in range(rows):
        line = ""
        for col in range(cols):
            idx = col * rows + row
            if idx < len(available_models):
                line += f" - {available_models[idx]}".ljust(25)
        print(line)
    
    model_choice = input("\nEnter the model name to download (e.g., yolov8n.pt): ")
    if model_choice in YOLO_MODELS:
        # Download to models directory
        destination = os.path.join(models_dir, model_choice)
        if download_file(YOLO_MODELS[model_choice], destination):
            print(f"Downloaded {model_choice} to {destination} successfully!")
            
            # Try to load and display classes for the downloaded model
            try:
                model = YOLO(destination)
                display_model_classes(model)
                return True
            except Exception as e:
                print(f"Could not display model classes: {e}")
                return False
        else:
            print(f"Failed to download {model_choice}")
            return False
    else:
        print(f"Invalid model name: {model_choice}")
        print("Please choose one of the available models listed above.")
        return False

def display_model_classes(model):
    """
    Display all classes in the model, highlighting luggage-related classes
    """
    print("\nAvailable classes in the model:")
    class_names = model.model.names
    
    # Get indices sorted by class names
    sorted_indices = sorted(range(len(class_names)), key=lambda i: class_names[i])
    
    # Display classes in columns
    cols = 3
    rows = (len(sorted_indices) + cols - 1) // cols
    
    for row in range(rows):
        line = ""
        for col in range(cols):
            idx = col * rows + row
            if idx < len(sorted_indices):
                i = sorted_indices[idx]
                name = class_names[i]
                # Highlight luggage classes
                if name.lower() in LUGGAGE_CLASSES:
                    line += f"{i}: \033[1m\033[92m{name}\033[0m".ljust(25)
                else:
                    line += f"{i}: {name}".ljust(25)
        print(line)
    
    print("\nLuggage-related classes are highlighted in green.")

def ensure_models_directory():
    """
    Check if 'models' directory exists and create it if not
    """
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    if not os.path.exists(models_dir):
        print(f"Creating models directory at: {models_dir}")
        os.makedirs(models_dir)
    else:
        print(f"Models directory exists at: {models_dir}")
    
    return models_dir

def download_specific_model(model_name):
    """
    Download a specific YOLO model
    
    Args:
        model_name: Name of the model to download (e.g., 'yolo11n.pt')
    
    Returns:
        bool: True if successful, False otherwise
    """
    models_dir = ensure_models_directory()
    model_path = os.path.join(models_dir, model_name)
    
    # Check if model already exists and is valid
    if os.path.exists(model_path):
        file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        if file_size_mb > 1:  # Most YOLO models are several MB in size
            print(f"Model already exists at: {model_path} ({file_size_mb:.1f} MB)")
            # Verify the model by trying to load it
            try:
                model = YOLO(model_path)
                print(f"Verified existing model {model_name} is valid")
                return True
            except Exception as e:
                print(f"Warning: Existing model file may be corrupt: {e}")
                print("Will attempt to re-download...")
                # Continue with download attempts
        else:
            print(f"Warning: Existing model file is too small ({file_size_mb:.1f} MB), may be corrupt")
            print("Will attempt to re-download...")
            
    # Try to download via Ultralytics API
    try:
        print(f"Attempting to download {model_name} via Ultralytics API...")
        model = YOLO(model_name)
        # Copy the model to our models directory
        source_path = model.ckpt_path
        print(f"Copying from {source_path} to {model_path}")
        import shutil
        shutil.copy2(source_path, model_path)
        print(f"Successfully downloaded {model_name} to {model_path}")
        return True
    except Exception as e:
        print(f"Error downloading {model_name} via API: {e}")
        
    # If API fails, try direct download from multiple potential sources
    try:
        # Get all available model download URLs
        base_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
        
        # Use the models dictionary from the main part of the script
        # which has been updated with more model versions        # If the model wasn't found in our dictionary, try to construct the URL
        if model_name not in YOLO_MODELS:
            # Try to infer the URL from the model name pattern
            if model_name.startswith("yolov"):
                # Format: yolovXY.pt where X is version and Y is size (n,s,m,l,x)
                inferred_url = base_url + model_name
                print(f"Model URL not predefined. Using inferred URL: {inferred_url}")
                success = download_file(inferred_url, model_path)
                return success
            else:
                print(f"No direct download URL available for {model_name}")
                print(f"Available models: {', '.join(YOLO_MODELS.keys())}")
                return False
        else:
            # Use predefined URL
            url = YOLO_MODELS[model_name]
            success = download_file(url, model_path)
            
            # Verify downloaded model
            if success:
                try:
                    # Try loading the model to verify it's valid
                    model = YOLO(model_path)
                    print(f"Successfully verified downloaded model {model_name}")
                    return True
                except Exception as e:
                    print(f"Downloaded model failed verification: {e}")
                    if os.path.exists(model_path):
                        os.remove(model_path)
                    return False
            return success
    except Exception as e:
        print(f"Error during direct download of {model_name}: {e}")
        if os.path.exists(model_path):
            os.remove(model_path)
        return False

if __name__ == "__main__":
    # Make sure the models directory exists
    models_dir = ensure_models_directory()
    
    # List of YOLOv10-v11 models to check
    v10_v11_models = ['yolov10n.pt', 'yolov10s.pt', 'yolov10m.pt',
                      'yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt']
    
    # Check which models are missing
    missing_models = []
    existing_models = []
    for model_name in v10_v11_models:
        model_path = os.path.join(models_dir, model_name)
        if os.path.exists(model_path) and os.path.getsize(model_path) > 1000000:
            existing_models.append(model_name)
        else:
            missing_models.append(model_name)
    
    # Report findings
    if existing_models:
        print(f"Found {len(existing_models)} YOLOv10-v11 models:")
        for model in existing_models:
            model_path = os.path.join(models_dir, model)
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f" - {model} ({size_mb:.1f} MB)")
    
    if not missing_models:
        print("\nAll YOLOv10-v11 models are already downloaded.")
    else:
        print(f"\nMissing {len(missing_models)} YOLOv10-v11 models: {', '.join(missing_models)}")
        print("Downloading missing models...")
        
        # Download each missing model
        for model_name in missing_models:
            print(f"\nDownloading {model_name}...")
            download_specific_model(model_name)
