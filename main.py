import io
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from typing import Optional
import os

# Import các hàm từ module image_to_text đã tái cấu trúc
from image_to_text import (
    get_device, load_model, load_tokenizer,
    extract_text_from_image, parse_json_response,
    DEFAULT_QUESTION,
)

# Global variables for model, tokenizer, and device
model = None
tokenizer = None
device = None

app = FastAPI(
    title="Image to Text API",
    description="API for processing images and extracting text information",
    version="1.0.0"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize model and tokenizer when the server starts"""
    global model, tokenizer, device

    # Sử dụng hàm get_device() từ image_to_text.py
    device = get_device()

    print("Loading model and tokenizer...")

    # Sử dụng các hàm tải model và tokenizer từ image_to_text.py
    model = load_model(device=device)
    tokenizer = load_tokenizer()

    print("Model and tokenizer loaded successfully!")


@app.post("/process-image/")
async def process_image(
    file: UploadFile = File(...),
    question: Optional[str] = DEFAULT_QUESTION
):
    """
    Upload an image and receive extracted text information as JSON.

    - **file**: The image file to upload
    - **question**: Optional question to guide the extraction (default extracts ID card information)
    """
    try:
        # Verify file is an image
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400, detail="File must be an image")

        # Read the image file
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')

        # Gọi hàm extract_text_from_image và nhận cả hai giá trị trả về
        response, processing_time = extract_text_from_image(
            image,
            model,
            tokenizer,
            device,
            question
        )

        # Tạo cấu trúc JSON chuẩn cho response
        json_response = {
            "result": parse_json_response(response),
            "processing_time": f"{processing_time:.2f} seconds"
        }

        return json_response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Return API information"""
    return {
        "message": "Image to Text API is running",
        "usage": "POST an image to /process-image/ endpoint"
    }


if __name__ == "__main__":
    import os
    # Lấy PORT từ biến môi trường nếu có (cần thiết cho các dịch vụ như Render)
    # Nếu không có biến môi trường PORT, sử dụng port 8000 mặc định
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
