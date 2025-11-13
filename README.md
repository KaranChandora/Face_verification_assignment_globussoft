#  Face Verification API (OpenCV)

This project implements a **Face Verification API** using **FastAPI** and **OpenCV’s Deep Neural Network (DNN)** module.  
It detects faces in two uploaded images, extracts lightweight pixel embeddings, and compares them using **cosine similarity** to check if both faces belong to the same person.

---

##  Features
- Face detection using **OpenCV DNN (ResNet-SSD model)**
- Compare two face images via **cosine similarity**
- Lightweight, no GPU or cloud dependency
- REST API built with **FastAPI**
- Returns similarity score and bounding boxes

---

##  Install Dependencies
1. fastapi 
2. uvicorn
3. opencv-python
4. numpy

---
## Model Files

A folder called models/ in your project and with these two files inside it:

- deploy.prototxt
Download: https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt

- res10_300x300_ssd_iter_140000.caffemodel
Download: https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/res10_300x300_ssd_iter_140000.caffemodel

## Commands

- In terminal, run - pip install -r requirements.txt (It will install all the dependencies)
- In terminal, run - uvicorn main:app --reload (it runs the 'app' object inside the main.py file)
- Now open your browser at: http://127.0.0.1:8000/docs

- You’ll see the FastAPI Swagger UI where you can upload two images and test the endpoint.

## API Endpoint
POST /verify

Description: Compares two uploaded face images and returns whether they belong to the same person.

Request (multipart/form-data):

| Field | Type | Description       |
| ----- | ---- | ----------------- |
| file1 | file | First face image  |
| file2 | file | Second face image |


### Sample Response

## Successful Verification
``` json
{
  "verification_result": "same person",
  "similarity_score": 0.83,
  "face_1_boxes": [[121, 177, 74, 82]],
  "face_2_boxes": [[203, 69, 58, 148]]
}
```

## If no faces are detected:
```
{
  "error": "No faces detected in one or both images"
}
```
## How It Works 

Reads both uploaded images and converts them to NumPy arrays. Detects faces using OpenCV’s pretrained DNN model. Extracts face regions, resizes them to 128×128, and flattens into embeddings. Computes cosine similarity between both embeddings. Returns "same person" if similarity > 0.75, else "different person".

## Notes

Works best on clear, frontal face images. Accuracy is limited since it uses pixel embeddings instead of deep features. For production, consider FaceNet, ArcFace, or DeepFace for higher accuracy.


## Example Output 
| Input                         | Output                            |
| ----------------------------- | --------------------------------- |
| Two photos of the same person |  `same person` (0.85+)            |
| Different people              |  `different person` (0.5 or less) |

