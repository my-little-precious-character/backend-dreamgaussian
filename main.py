import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
import os
import shutil
from typing import Dict
from uuid import uuid4
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

######## type ########

class TaskType(str, Enum):
    TEXT_TO_3D = "text_to_3d"
    IMAGE_TO_3D = "image_to_3d"
    TEXT_TO_3D_TEST = "text_to_3d_test"
    IMAGE_TO_3D_TEST = "image_to_3d_test"

@dataclass
class TaskItem:
    id: str
    type: TaskType
    data: dict

######## env var ########

# Load .env
load_dotenv()

SAMPLE_DIR = "results-sample"

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

RESULT_DIR = os.getenv("RESULT_DIR", "results")
os.makedirs(RESULT_DIR, exist_ok=True)

######## global variables ########

# queue & task
task_queue = asyncio.Queue()
task_progress: Dict[str, str] = {} # [task_id, queued | processing | done]
task_result_paths: Dict[str, str] = {}  # [task_id, file path]

######## worker ########

async def handle_test(task):
    for i in range(100):
        await asyncio.sleep(0.01)
        task_progress[task.id] = f"processing ({(i + 1) * 1}%)"

    src_path = os.path.join(SAMPLE_DIR, "luigi_mesh.obj")
    dst_path = os.path.join(RESULT_DIR, f"{task.id}_mesh.obj")

    shutil.copyfile(src_path, dst_path)

    task_result_paths[task.id] = dst_path # FIXME:

@asynccontextmanager
async def lifespan(app: FastAPI):
    async def worker():
        while True:
            task: TaskItem = await task_queue.get()
            task_progress[task.id] = "processing"
            try:
                if task.type == TaskType.TEXT_TO_3D:
                    pass    # TODO: 실제 처리
                elif task.type == TaskType.IMAGE_TO_3D:
                    pass    # TODO: 실제 처리
                elif task.type in (TaskType.TEXT_TO_3D_TEST, TaskType.IMAGE_TO_3D_TEST):
                    await handle_test(task)
                task_progress[task.id] = "done"
            except Exception as e:
                task_progress[task.id] = f"error: {str(e)}"

    # Run wordker
    asyncio.create_task(worker())

    yield

######## fastapi ########

app = FastAPI(lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "https://my-character.cho0h5.org",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "welcome"}

@app.post("/text-to-3d")
async def text_to_3d(prompt: str = Form(...), mode: str = "prod"):
    task_id = uuid4().hex
    task_type = TaskType.TEXT_TO_3D if mode == "prod" else TaskType.TEXT_TO_3D_TEST
    task = TaskItem(id=task_id, type=task_type, data={"prompt": prompt})
    await task_queue.put(task)
    task_progress[task_id] = "queued"
    return {"task_id": task_id}

@app.post("/image-to-3d")
async def upload_image(file: UploadFile = File(...), mode: str = "prod"):
    # Check if the uploaded file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    # Generate filename
    task_id = uuid4().hex
    file_extension = os.path.splitext(file.filename)[1]
    file_path = os.path.join(UPLOAD_DIR, f"{task_id}{file_extension}")

    # Store file
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    task_type = TaskType.IMAGE_TO_3D if mode == "prod" else TaskType.IMAGE_TO_3D_TEST
    task = TaskItem(id=task_id, type=task_type, data={"image_path": file_path})
    await task_queue.put(task)
    task_progress[task_id] = "queued"

    # Response
    return {"task_id": task_id}

@app.websocket("/text-to-3d/ws")
@app.websocket("/image-to-3d/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        task_id = await websocket.receive_text()

        while True:
            await asyncio.sleep(0.01)
            status = task_progress.get(task_id, "unknown")

            await websocket.send_text(f"status: {status}")

            if status == "done" or status.startswith("error"):
                break;

    finally:
        await websocket.close()

@app.get("/text-to-3d")
@app.get("/image-to-3d")
async def get_image_result(task_id: str):
    if task_progress.get(task_id) != "done":
        raise HTTPException(status_code=400, detail="Task not complete")

    path = task_result_paths.get(task_id)
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(path, media_type="application/octet-stream", filename=os.path.basename(path))
