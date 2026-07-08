import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
# Các hàm suy luận.
from test import video_stream, image

# Ứng dụng FastAPI.
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

# Route kiểm tra trạng thái.
@app.get("/")
def read_root():
    return FileResponse("frontend/main.html")

@app.post("/predict/video")
def run_video_stream():
    # Dùng model đã được nạp khi import.
    results = video_stream()
    return {"status": "success", "results": str(results)}

@app.post("/predict/image")
async def run_image_prediction(uploaded_image: UploadFile = File(...)):
    # Dùng model đã được nạp khi import.
    suffix = Path(uploaded_image.filename or "image.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(await uploaded_image.read())
        temp_path = temp_file.name

    results = image(temp_path)
    return {"status": "success", "results": str(results)}

# Chạy trực tiếp khi mở file.
if __name__ == "__main__":
    uvicorn.run("API:app", host="127.0.0.1", port=8000, reload=True)