import os
from uuid import uuid4
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

######## environment variables ########

# Load .env
load_dotenv()

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

######## fastapi ########

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "welcome"}

@app.post("/image-to-3d/test")
async def upload_image(file: UploadFile = File(...)):
    # Check if the uploaded file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    # Generate filename
    file_extension = os.path.splitext(file.filename)[1]
    filename = f"{uuid4().hex}{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    # Store file
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    # Response
    return JSONResponse(
        content={"filename": filename, "message": "File uploaded successfully."}
    )
