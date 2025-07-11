from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import io
import torch
from torchvision import models, transforms
from PIL import Image

app = FastAPI(
    title="VOC Classifier API",
    description="Upload an image to /predict and get back the VOC class prediction",
    version="1.0",
)

# 0) Root & health endpoints
@app.get("/")
def read_root():
    return {"message": "VOC Classifier API is running. POST to /predict"}

@app.get("/health")
def health():
    return {"status": "ok"}

# 1) Model loading
MODEL_PATH = "model/best_efficientnet_b0.pth"
VOC_CLASSES = [
    'background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat',
    'chair','cow','diningtable','dog','horse','motorbike','person','pottedplant',
    'sheep','sofa','train','tvmonitor'
]

def load_effnet():
    m = models.efficientnet_b0(pretrained=False)
    in_feats = m.classifier[1].in_features
    m.classifier[1] = torch.nn.Linear(in_feats, len(VOC_CLASSES)-1)
    state = torch.load(MODEL_PATH, map_location="cpu")
    m.load_state_dict(state)
    m.eval()
    return m

model = load_effnet()

preproc = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# 2) Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(("png","jpg","jpeg")):
        raise HTTPException(415, "Unsupported file type")
    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except:
        raise HTTPException(400, "Invalid image")
    inp = preproc(img).unsqueeze(0)
    with torch.no_grad():
        out = model(inp)
        idx = int(out.argmax(1))
    return JSONResponse({
        "filename": file.filename,
        "predicted_class": VOC_CLASSES[idx+1]
    })
