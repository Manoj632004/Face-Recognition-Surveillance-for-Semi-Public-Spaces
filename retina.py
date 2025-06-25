import cv2
import numpy as np
import os
from openvino.runtime import Core
import re
import shutil
from collections import deque

os.environ["INTEL_OPENVINO_TARGET"] = "GPU"

def preprocess_face(face_img):
    resized = cv2.resize(face_img, (112, 112))
    chw = resized.transpose((2, 0, 1))
    input_tensor = np.expand_dims(chw, axis=0).astype(np.float32)
    return input_tensor

def preprocess_face_detection(frame, w, h):
    resized = cv2.resize(frame, (w, h))
    blob = resized.transpose((2, 0, 1))
    blob = np.expand_dims(blob, axis=0).astype(np.float32)
    return blob

def cosine_similarity(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    return float(np.dot(a_norm, b_norm))

def enhance_low_light_image(frame):
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    
    gamma = 2.0
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    frame_enhanced = cv2.LUT(frame, table)
    
    lab = cv2.cvtColor(frame_enhanced, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    ycrcb = cv2.cvtColor(final, cv2.COLOR_BGR2YCrCb)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    final = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    
    final = cv2.bilateralFilter(final, 9, 300, 300)
    
    return final

def handle_long_distance(frame):
    scaled = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return scaled

def load_known_faces_embeddings(ie, arcface_compiled, arcface_output_layer, known_faces_dir):
    known_embeddings = {}
    print("Loading known faces embeddings...")
    for foldername in os.listdir(known_faces_dir):
        folder_path = os.path.join(known_faces_dir, foldername)
        if os.path.isdir(folder_path):
            embeddings = []
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    path = os.path.join(folder_path, filename)
                    img = cv2.imread(path)
                    if img is None:
                        print(f"Warning: Could not read image {path}")
                        continue
                    face_input = preprocess_face(img)
                    result = arcface_compiled([face_input])[arcface_output_layer]
                    embedding = result[0]
                    embeddings.append(embedding)
            if embeddings:
                avg_embedding = np.mean(embeddings, axis=0)
                known_embeddings[foldername] = avg_embedding
                print(f"Loaded embedding for {foldername} (avg of {len(embeddings)} images)")
    return known_embeddings

def get_next_face_id(known_faces_dir):
    existing_ids = set()
    pattern = re.compile(r'^face_(\d{1,2})$')
    for foldername in os.listdir(known_faces_dir):
        if pattern.match(foldername):
            match = pattern.match(foldername)
            if match:
                face_id = int(match.group(1))
                existing_ids.add(face_id)
    face_id = 1
    while face_id in existing_ids:
        face_id += 1
    return face_id

def create_new_known_face_folder(base_known_dir):
    face_id = get_next_face_id(base_known_dir)
    folder_name = f"face_{face_id}"
    folder_path = os.path.join(base_known_dir, folder_name)
    os.makedirs(folder_path, exist_ok=False)
    print(f"Created new known face folder: {folder_path}")
    return folder_path, folder_name

def delete_folder(folder_path):
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        shutil.rmtree(folder_path)
        print(f"Deleted folder: {folder_path}")

def is_same_face(embedding1, embedding2, threshold=0.7):
    return cosine_similarity(embedding1, embedding2) >= threshold

def process_frame(ie, arcface_compiled, arcface_output_layer, face_det_compiled,
                  face_det_input_layer, face_det_output_layer, fd_input_h, fd_input_w,
                  known_embeddings, frame, unknown_buffer, unknown_images,
                  SIMILARITY_THRESHOLD, ENROLL_CAPTURE_COUNT, DUPLICATE_SIMILARITY_THRESHOLD,
                  known_faces_dir): 
    
    enhanced_frame = enhance_low_light_image(frame)
    
    scaled_frame = handle_long_distance(enhanced_frame)
    
    face_scales = [1.0, 0.8, 1.2, 1.4]
    faces = []
    
    for scale in face_scales:
        width = int(scaled_frame.shape[1] * scale)
        height = int(scaled_frame.shape[0] * scale)
        resized = cv2.resize(scaled_frame, (width, height))
        
        input_blob = preprocess_face_detection(resized, fd_input_w, fd_input_h)
        det_result = face_det_compiled([input_blob])[face_det_output_layer]
        detections = det_result[0][0]
        
        for detection in detections:
            confidence = float(detection[2])
            if confidence > 0.3:  
                x_min = int(detection[3] * width)
                y_min = int(detection[4] * height)
                x_max = int(detection[5] * width)
                y_max = int(detection[6] * height)
                
                x_min = int(x_min / scale)
                y_min = int(y_min / scale)
                x_max = int(x_max / scale)
                y_max = int(y_max / scale)
                
                x_min = max(0, min(x_min, scaled_frame.shape[1] - 1))
                y_min = max(0, min(y_min, scaled_frame.shape[0] - 1))
                x_max = max(0, min(x_max, scaled_frame.shape[1] - 1))
                y_max = max(0, min(y_max, scaled_frame.shape[0] - 1))
                
                faces.append((x_min, y_min, x_max, y_max))
    
    unique_faces = []
    seen = set()
    for face in faces:
        key = (face[0], face[1], face[2], face[3])
        if key not in seen:
            seen.add(key)
            unique_faces.append(face)
    
    faces_detected = False
    for (x_min, y_min, x_max, y_max) in unique_faces:
        cv2.rectangle(scaled_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        faces_detected = True
        
        face_roi = scaled_frame[y_min:y_max, x_min:x_max]
        if face_roi.size == 0:
            continue
        
        arcface_input = preprocess_face(face_roi)
        arcface_result = arcface_compiled([arcface_input])[arcface_output_layer]
        embedding = arcface_result[0]
        
        best_match_name = None
        best_similarity = -1
        
        for name, known_emb in known_embeddings.items():
            sim = cosine_similarity(embedding, known_emb)
            if sim > best_similarity:
                best_similarity = sim
                best_match_name = name
        
        if best_similarity >= SIMILARITY_THRESHOLD:
            label = f"{best_match_name} ({best_similarity*100:.1f}%)"
        else:
            label = "Unknown"
            unknown_buffer.append(embedding)
            unknown_images.append(face_roi)
            
            if len(unknown_buffer) >= ENROLL_CAPTURE_COUNT:
                is_duplicate = False
                max_sim_to_known = 0.0
                
                for name, known_emb in known_embeddings.items():
                    current_sim = cosine_similarity(embedding, known_emb)
                    if current_sim > DUPLICATE_SIMILARITY_THRESHOLD:
                        is_duplicate = True
                        break
                    if current_sim > max_sim_to_known:
                        max_sim_to_known = current_sim
                
                if not is_duplicate:
                    sims = []
                    for i in range(len(unknown_buffer)):
                        for j in range(i + 1, len(unknown_buffer)):
                            sims.append(cosine_similarity(unknown_buffer[i], unknown_buffer[j]))
                    avg_sim = sum(sims) / len(sims) if sims else 1.0
                    
                    if avg_sim > 0.8:
                        new_face_folder, folder_name = create_new_known_face_folder(known_faces_dir)
                        for idx, img in enumerate(unknown_images):
                            img_path = os.path.join(new_face_folder, f"{idx + 1}.jpg")
                            cv2.imwrite(img_path, img)
                        print(f"New face saved as known in folder: {new_face_folder}")
                        new_embedding = np.mean(unknown_buffer, axis=0)
                        known_embeddings[folder_name] = new_embedding
                        print(f"Updated known embeddings with new face: {folder_name}")
                    else:
                        print("Not enough consistency in captures. Captures will not be saved.")
                else:
                    print(f"Duplicate face detected with similarity {max_sim_to_known:.2f}. Enrollment prevented.")
                
                unknown_buffer.clear()
                unknown_images.clear()
        
        cv2.putText(scaled_frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

    return scaled_frame, faces_detected

def main():
    ie = Core()
    
    face_det_model_path = "face-detection-0200.xml"
    arcface_model_path = "face-recognition-resnet100-arcface-onnx.xml"
    device_name = "GPU"
    
    face_det_model = ie.read_model(face_det_model_path)
    face_det_compiled = ie.compile_model(face_det_model, device_name)
    face_det_input_layer = face_det_compiled.input(0)
    face_det_output_layer = face_det_compiled.output(0)
    
    arcface_model = ie.read_model(arcface_model_path)
    arcface_compiled = ie.compile_model(arcface_model, device_name)
    arcface_input_layer = arcface_compiled.input(0)
    arcface_output_layer = arcface_compiled.output(0)
    
    fd_input_h = face_det_input_layer.shape[2]
    fd_input_w = face_det_input_layer.shape[3]
    
    faces_dir = "faces"
    known_faces_dir = os.path.join(faces_dir, "known")
    
    os.makedirs(known_faces_dir, exist_ok=True)
    
    known_embeddings = load_known_faces_embeddings(ie, arcface_compiled, arcface_output_layer, known_faces_dir)
    print(f"Total known faces loaded: {len(known_embeddings)}")
    
    SIMILARITY_THRESHOLD = 0.7
    ENROLL_CAPTURE_COUNT = 50
    DUPLICATE_SIMILARITY_THRESHOLD = 0.85
    
    unknown_buffer = deque(maxlen=ENROLL_CAPTURE_COUNT)
    unknown_images = []
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        processed_frame, faces_detected = process_frame(ie, arcface_compiled, arcface_output_layer, face_det_compiled,
                                                        face_det_input_layer, face_det_output_layer, fd_input_h, fd_input_w,
                                                        known_embeddings, frame, unknown_buffer, unknown_images,
                                                        SIMILARITY_THRESHOLD, ENROLL_CAPTURE_COUNT, DUPLICATE_SIMILARITY_THRESHOLD,
                                                        known_faces_dir)  
        
        display_frame = cv2.resize(processed_frame, (640, 480))
        
        cv2.imshow("Webcam Feed", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
