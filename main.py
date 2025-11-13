from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import cv2

app = FastAPI(title="Face Verification API (OpenCV)")

# Paths to face detection model
prototxt_path = "models/deploy.prototxt"
model_path = "models/res10_300x300_ssd_iter_140000.caffemodel"

# Load the pre-trained OpenCV DNN model
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)


def read_image(file: UploadFile):
    #Reads uploaded image and converts it to a numpy RGB array.
    file_bytes = np.asarray(bytearray(file.file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def get_face_embeddings(img, confidence_threshold=0.4):
    #Detect faces and create simple flattened pixel embeddings.
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    embeddings = []
    boxes = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            # Clip coordinates to image boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Extract and preprocess face region
            face = img[y1:y2, x1:x2]
            if face.size == 0:
                continue
            face = cv2.resize(face, (128, 128))
            face_flat = face.flatten() / 255.0  # simple embedding
            embeddings.append(face_flat)
            boxes.append([x1, y1, x2 - x1, y2 - y1])

    return np.array(embeddings), boxes


def cosine_similarity(a, b):
    #Computes cosine similarity between two embeddings.
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


@app.post("/verify")
async def verify_faces(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    #API endpoint for verifying if two faces belong to the same person.
    img1 = read_image(file1)
    img2 = read_image(file2)

    emb1, boxes1 = get_face_embeddings(img1)
    emb2, boxes2 = get_face_embeddings(img2)

    print(f"Faces detected: {len(emb1)} in image1, {len(emb2)} in image2")

    if len(emb1) == 0 or len(emb2) == 0:
        return JSONResponse(
            status_code=400,
            content={"error": "No faces detected in one or both images"}
        )

    # Compare first face from each image
    similarity = cosine_similarity(emb1[0], emb2[0])
    threshold = 0.75  # adjust based on your testing
    result = "same person" if similarity > threshold else "different person"

    return {
        "verification_result": result,
        "similarity_score": float(similarity),
        "face_1_boxes": [[int(x) for x in box] for box in boxes1],
        "face_2_boxes": [[int(x) for x in box] for box in boxes2]
    }






