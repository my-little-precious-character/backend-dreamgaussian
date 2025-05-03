import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
import os
from typing import Dict
from uuid import uuid4
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

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

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

RESULT_DIR = os.getenv("RESULT_DIR", "results")
os.makedirs(RESULT_DIR, exist_ok=True)

######## global variables ########

# queue & task
task_queue = asyncio.Queue()
task_progress: Dict[str, str] = {} # [task_id, queued | processing | done]
task_result_paths: Dict[str, str] = {}  # [task_id, file path]
ws_connections: Dict[str, WebSocket] = {}

######## worker ########

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
                    for i in range(5):
                        await asyncio.sleep(1)
                        task_progress[task.id] = f"processing ({(i + 1) * 20}%)"
                    task_result_paths[task.id] = "results/sample.png.obj" # FIXME:
                task_progress[task.id] = "done"
            except Exception as e:
                task_progress[task.id] = f"error: {str(e)}"
            finally:
                if task.id in ws_connections:
                    await ws_connections[task.id].send_text(f"status: {task_progress[task.id]}")
                    await ws_connections[task.id].close()
                    ws_connections.pop(task.id, None)

    # Run wordker
    asyncio.create_task(worker())

    yield

######## fastapi ########

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {"message": "welcome"}

@app.post("/image-to-3d")
async def upload_image(file: UploadFile = File(...), mode: str = "prod"):
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

    task_id = uuid4().hex
    task_type = TaskType.IMAGE_TO_3D if mode == "prod" else TaskType.IMAGE_TO_3D_TEST
    task = TaskItem(id=task_id, type=task_type, data={"image_path": file_path})
    await task_queue.put(task)
    task_progress[task_id] = "queued"

    # Response
    return {"task_id": task_id}

@app.websocket("/image-to-3d/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        task_id = await websocket.receive_text()
        ws_connections[task_id] = websocket
        await websocket.send_text(f"status: {task_progress.get(task_id, 'unknown')}")
        while task_progress.get(task_id) != "done":
            await asyncio.sleep(1)
            await websocket.send_text(f"status: {task_progress[task_id]}")
    except WebSocketDisconnect:
        ws_connections.pop(task_id, None)

@app.get("/image-to-3d")
async def get_image_result(task_id: str):
    if task_progress.get(task_id) != "done":
        raise HTTPException(status_code=400, detail="Task not complete")

    path = task_result_paths.get(task_id)
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(path, media_type="application/octet-stream", filename=os.path.basename(path))
