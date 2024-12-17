from fastapi import FastAPI
from ultralytics import YOLO

app = FastAPI()


@app.get("/")
async def home():
   return {"data": "Hello World"}


@app.get("/predict")
async def get_model():
    model = YOLO("yolo11n.pt")  # pretrained YOLO11n model
    
    results = model(["image_1.jpg",])  # return a list of Results objects
    
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        result.show()  # display to screen
        result.save(filename="result.jpg")  # save to disk
    
    return {'model_predict': True}

   
