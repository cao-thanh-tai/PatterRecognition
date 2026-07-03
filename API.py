# main.py
from fastapi import FastAPI
import uvicorn
# Import các hàm xử lý từ file tách riêng của bạn
# (Giả sử bạn đã tạo file models_predictor.py như ở bước trước)
from test import video_stream, image

# 1. Khởi tạo FastAPI app
app = FastAPI()

# 2. Khai báo một API Endpoint (Route) mẫu
@app.get("/")
def read_root():
    return {"message": "Server FastAPI đã sẵn sàng, các model đã load xong trên RAM!"}

@app.post("/predict/video")
def run_video_stream():
    # Hàm này gọi model cực nhanh vì model đã được nạp ở scope global khi import
    results = video_stream()
    return {"status": "success", "results": str(results)}

@app.post("/predict/image")
def run_image_prediction(image_path: str):
    # Hàm này gọi model cực nhanh vì model đã được nạp ở scope global khi import
    results = image(image_path)
    return {"status": "success", "results": str(results)}

# 3. Đoạn code để bạn có thể bấm "Run" trực tiếp file .py này trong VS Code
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)