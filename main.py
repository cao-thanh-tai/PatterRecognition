from test import video_stream, image
import uvicorn


if __name__ == "__main__":
    uvicorn.run("API:app", host="127.0.0.1", port=8000, reload=True)    