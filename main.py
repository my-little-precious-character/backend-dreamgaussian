import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
import os
import shutil
import subprocess
from typing import Dict, Optional
from uuid import uuid4
from fastapi import FastAPI, File, Form, Query, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import shlex
import shutil
import torch

import logging
import time

import subprocess

from PIL import Image, ImageFilter
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import cv2
import numpy as np


######## make log ########

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()  # 콘솔 출력도 유지
    ]
)

logger = logging.getLogger(__name__)

######## type ########

class TaskType(str, Enum):
    TEXT_TO_3D = "text_to_3d"
    IMAGE_TO_3D = "image_to_3d"
    TEXT_TO_3D_TEST = "text_to_3d_test"
    IMAGE_TO_3D_TEST = "image_to_3d_test"

class FileType(str, Enum):
    obj = "obj"
    mtl = "mtl"
    albedo = "albedo"

@dataclass
class TaskItem:
    id: str
    type: TaskType
    data: dict

######## env var ########

# Load .env
load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DREAMGAUSSIAN_DIR = os.path.join(BASE_DIR, "..", "dreamgaussian")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", os.path.join(BASE_DIR, "uploads"))
RESULT_DIR = os.getenv("RESULT_DIR", os.path.join(BASE_DIR, "results"))

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# print(UPLOAD_DIR)
# print(RESULT_DIR)

SAMPLE_DIR = "results-sample"


######## upscaling model ########
device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")


net_rrdb = RRDBNet(num_in_ch=3, num_out_ch=3,
                   num_feat=64, num_block=23,
                   num_grow_ch=32, scale=4)

model = RealESRGANer(
    scale=4,
    model_path='weights/RealESRGAN_x4plus.pth',
    dni_weight=None,
    model=net_rrdb,          # ★ network 대신 model 파라미터
    tile=0, tile_pad=10, pre_pad=0,
    half=True,
    device=device,
)


######## global variables ########

# queue & task
task_queue = asyncio.Queue()
task_progress: Dict[str, str] = {} # [task_id, queued | processing | done | error]
task_result_paths: Dict[str, str] = {}  # [task_id, directory path]


#### upscaling ####
async def upscale_image(task_id: str, input_path: str):
    loop = asyncio.get_running_loop()

    def _work() -> None:
        with Image.open(input_path) as im:
            logger.info(f"task_id: {task_id}, input image size: {im.size}") 
            has_alpha = ("A" in im.getbands()) or ("transparency" in im.info)
            if has_alpha:
                alpha = im.convert("RGBA").getchannel("A")   # tRNS도 반영
            rgb   = im.convert("RGB")

            sr_bgr, _ = model.enhance(cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR), outscale=4)
            sr_rgb = Image.fromarray(cv2.cvtColor(sr_bgr, cv2.COLOR_BGR2RGB))
            
            if has_alpha:
                sr_a  = alpha.resize(sr_rgb.size, Image.BICUBIC)
                sr    = Image.merge("RGBA", (*sr_rgb.split(), sr_a))
            else:
                sr = sr_rgb

            sr.save(input_path, compress_level=2)  # PNG 덮어쓰기
            logger.info(f"task_id: {task_id}, output image size: {sr.size}") 


    await loop.run_in_executor(None, _work)
    logger.info(f"task_id: {task_id}, Upscaling complete! Filename: {input_path}") 


#### convert to png ####
async def convert_to_png(task_id: str, input_path):
    if input_path.lower().endswith('.png'):
        return input_path
    output_path = os.path.splitext(input_path)[0] + '.png'

    try:
        with Image.open(input_path) as img:
            img = img.convert('RGB')
            # img = img.filter(ImageFilter.MedianFilter(size=3))
            # img = img.convert('RGBA') # also need transparency
            img.save(output_path, 'PNG')
            logger.info(f"task_id: {task_id}, Converted: {input_path} → {output_path}")
            return output_path
    except Exception as e:
       logger.info(f"task_id: {task_id}, Convert failed: {e}")
       return None


#### test ####
async def handle_test(task) -> Optional[str]:
    def copy(src, dst):
        src_path = os.path.join(SAMPLE_DIR, src)
        dst_path = os.path.join(result_path, dst)
        shutil.copyfile(src_path, dst_path)
        return dst_path

    logger.info(f"task_id: {task.id}, function:s handle_test")
    result_path = os.path.join(RESULT_DIR, task.id)
    os.makedirs(result_path, exist_ok=True)
    
    # Wait for 1 second
    # for i in range(100):
    #     await asyncio.sleep(0.01)
    #     task_progress[task.id] = f"processing ({(i + 1) * 1}%)"

    # Copy dummy results
    mtl_filename = f"{task.id}_mesh.mtl"
    albedo_filename = f"{task.id}_mesh_albedo.png"
    obj_path = copy("luigi_mesh.obj", f"{task.id}_mesh.obj")
    mtl_path = copy("luigi_mesh.mtl", mtl_filename)
    copy("luigi_mesh_albedo.png", albedo_filename)

    # Use sed to fix mtllib in .obj
    subprocess.run([
        "sed", "-i",
        f"s|mtllib luigi_mesh.mtl|mtllib {mtl_filename}|g",
        obj_path
    ], check=True)

    # Use sed to fix map_Kd in .mtl
    subprocess.run([
        "sed", "-i",
        f"s|map_Kd luigi_mesh_albedo.png|map_Kd {albedo_filename}|g",
        mtl_path
    ], check=True)

    return result_path

######## text to 3d Stable Diffusion + image.yaml  ########
async def run_dreamgaussian_text_image(task_id: str, task_promt: dict[str, str], elevation: int = 0) -> Optional[str]:
    try:
        task_progress[task_id] = "processing"
        task_value = " ".join(task_promt.values())        
        logger.info(f"task_id: {task_id}, function: run_dreamgaussian_text, prompt: {task_value}")

        #### 여기에서 image_path로 image 생성해야함.
        image_path = os.path.join(UPLOAD_DIR, f"{task_id}.png")
        
        command_f = f"""
        python3 two_stage_controlnet.py \
        --control_image "./new_image.png" \
        --prompt "{task_value}, t-pose, full body, arms outstretched, facing forward, standing, no background, photorealistic, studio lighting, centered" \
        --negative "low quality, worst quality, multiple people, two faces, extra limbs, extra arms, extra legs, mutated hands, mutated legs, deformed body, merged faces, merged bodies, back_head, shadow, floor, cropped, duplicate, strange, distorted" \
        --output {image_path}"""


        process_f = await asyncio.create_subprocess_exec(
                *shlex.split(command_f),
                cwd=BASE_DIR,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
        while True:
            line = await process_f.stdout.readline()
            if not line:
                break
            logger.info(f"[{task_id}] {line.decode().strip()}")

            
        await process_f.wait()
        if process_f.returncode != 0:
            logger.info(f"[{task_id}] Error: {command} failed with code {process_f.returncode}")
            return None

        
        # png가 아니면 png로 바꿔주기
        image_path = await convert_to_png(task_id, image_path)
        if image_path is None:
            return None

        # 파일명 설정
        name, ext = os.path.splitext(os.path.basename(image_path)) 
        # name은 오직 파일 명
        # 입력: "/app/backend-dreamgaussian/uploads/luigi.png"
        # 출력: "luigi.png"
        processed_image = f"{name}_rgba.png"
        input_image_path = os.path.join(DREAMGAUSSIAN_DIR, "data", processed_image)
        output_dir = os.path.join(DREAMGAUSSIAN_DIR, "logs", "outputs")
        result_dir = os.path.join(RESULT_DIR, f"{task_id}")

        # 1. image preprocessing (process.py)
        process_command = f"python3 process.py {image_path}"
        process = await asyncio.create_subprocess_exec(
            *shlex.split(process_command),
            cwd=os.path.join(DREAMGAUSSIAN_DIR),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            logger.info(f"[{task_id}] process.py failed:\n{stderr.decode().strip()}")
            return None
        
        # image move (UPLOAD_DIR -> data/)
        if os.path.exists(input_image_path):
            os.remove(input_image_path)
        shutil.move(image_path, input_image_path)
        

        # stage 1 (main.py)
        command_1 = f"""
        python3 main.py \
          --config configs/image.yaml  \
          input=data/{processed_image} \
          save_path=outputs/{name}_mesh \
          elevation={elevation} \
          force_cuda_rast=True
        """

        # stage 2 (main2.py)
        command_2 = f"""
        python3 main2.py \
          --config configs/image.yaml  \
          input=data/{processed_image} \
          save_path=outputs/{name}_mesh \
          elevation={elevation} \
          force_cuda_rast=True
        """
        
        # 프로세스 실행
        for i, command in enumerate([command_1, command_2]):
            process = await asyncio.create_subprocess_exec(
                *shlex.split(command),
                cwd=DREAMGAUSSIAN_DIR,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                logger.info(f"[{task_id}] {line.decode().strip()}")

                
            await process.wait()
            if process.returncode != 0:
                logger.info(f"[{task_id}] Error: {command} failed with code {process.returncode}")
                return None
            
        
        task_progress[task_id] = "processing (100%)"   

        if os.path.exists(output_dir):
            target_files = [
                f"{name}_mesh.obj",
                f"{name}_mesh.mtl",
                f"{name}_mesh_albedo.png"
            ]

            if os.path.exists(result_dir):
                # 폴더 내부의 파일만 삭제
                for file in os.listdir(result_dir):
                    file_path = os.path.join(result_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
            else:
                # 폴더가 아예 없으면 생성
                os.makedirs(result_dir, exist_ok=True)
                    
            for file in target_files:
                src_file = os.path.join(output_dir, file)
                dst_file = os.path.join(result_dir, file)
                
                if os.path.isfile(src_file):
                    shutil.move(src_file, dst_file)
                    logger.info(f"Moved: {src_file} -> {dst_file}")
            
            dst_file = os.path.join(result_dir, target_files[2])
            await upscale_image(task_id, dst_file)

            logger.info(f"[{task_id}] Done: {result_dir}")
            return result_dir
        
        logger.info(f"[{task_id}] Error: failed to make {output_dir}")
        return None

    except Exception as e:
        logger.info(f"[{task_id}] Error: {e}")
        return None


######## text to 3d text.yaml ########
async def run_dreamgaussian_text(task_id: str, task_promt: dict[str, str], elevation: int = 0) -> Optional[str]:
    try:
        task_progress[task_id] = "processing"
        
        output_dir = os.path.join(DREAMGAUSSIAN_DIR, "logs", "outputs")
        result_dir = os.path.join(RESULT_DIR, f"{task_id}")
        
        task_value = " ".join(task_promt.values())
        logger.info(f"task_id: {task_id}, function: run_dreamgaussian_text, prompt: {task_value}")

        command_1 = f"""
        python3 main.py \
          --config configs/text.yaml  \
          prompt=\"{task_value}\" \
          save_path=outputs/{task_id}_mesh\
          elevation={elevation} \
          force_cuda_rast=True
        """
        command_2 = f"""
        python3 main2.py \
          --config configs/text.yaml  \
          prompt=\"{task_value}\" \
          save_path=outputs/{task_id}_mesh\
          elevation={elevation} \
          force_cuda_rast=True
        """
                
        # 프로세스 실행
        for i, command in enumerate([command_1, command_2]):
            process = await asyncio.create_subprocess_exec(
                *shlex.split(command),
                cwd=DREAMGAUSSIAN_DIR,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                logger.info(f"[{task_id}] {line.decode().strip()}")

                
            await process.wait()
            if process.returncode != 0:
                logger.info(f"[{task_id}] Error: {command} failed with code {process.returncode}")
                return None
            
        
        task_progress[task_id] = "processing (100%)"   


        if os.path.exists(output_dir):
            target_files = [
                f"{task_id}_mesh.obj",
                f"{task_id}_mesh.mtl",
                f"{task_id}_mesh_albedo.png"
            ]

            if os.path.exists(result_dir):
                # 폴더 내부의 파일만 삭제
                for file in os.listdir(result_dir):
                    file_path = os.path.join(result_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
            else:
                # 폴더가 아예 없으면 생성
                os.makedirs(result_dir, exist_ok=True)
                    
            for file in target_files:
                src_file = os.path.join(output_dir, file)
                dst_file = os.path.join(result_dir, file)
                
                if os.path.isfile(src_file):
                    shutil.move(src_file, dst_file)
                    logger.info(f"Moved: {src_file} -> {dst_file}")
            
            logger.info(f"[{task_id}] Done: {result_dir}")
            return result_dir
        
        logger.info(f"[{task_id}] Error: failed to make {output_dir}")
        return None

    except Exception as e:
        logger.info(f"[{task_id}] Error: {e}")
        return None



######## text to 3d text_mv.yaml(MVDream 멀티뷰 확산 모델) ########
async def run_dreamgaussian_text2(task_id: str, task_promt: dict[str, str], elevation: int = 0) -> Optional[str]:
    try:
        task_progress[task_id] = "processing"
        
        output_dir = os.path.join(DREAMGAUSSIAN_DIR, "logs", "outputs")
        result_dir = os.path.join(RESULT_DIR, f"{task_id}")
        
        task_value = " ".join(task_promt.values())        
        logger.info(f"task_id: {task_id}, function: run_dreamgaussian_text, prompt: {task_value}")


        command_1 = f"""
        python3 main.py \
          --config configs/imagedream.yaml  \
          prompt=\"{task_value}\" \
          save_path=outputs/{task_id}_mesh\
          elevation={elevation} \
          force_cuda_rast=True \
          train_geo=true
        """
        command_2 = f"""
        python3 main2.py \
          --config configs/imagedream.yaml  \
          prompt=\"{task_value}\" \
          save_path=outputs/{task_id}_mesh\
          elevation={elevation} \
          force_cuda_rast=True \
          train_geo=true
        """
                
        # 프로세스 실행
        for i, command in enumerate([command_1, command_2]):
            process = await asyncio.create_subprocess_exec(
                *shlex.split(command),
                cwd=DREAMGAUSSIAN_DIR,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                logger.info(f"[{task_id}] {line.decode().strip()}")

                
            await process.wait()
            if process.returncode != 0:
                logger.info(f"[{task_id}] Error: {command} failed with code {process.returncode}")
                return None
            
        
        task_progress[task_id] = "processing (100%)"   


        if os.path.exists(output_dir):
            target_files = [
                f"{task_id}_mesh.obj",
                f"{task_id}_mesh.mtl",
                f"{task_id}_mesh_albedo.png"
            ]

            if os.path.exists(result_dir):
                # 폴더 내부의 파일만 삭제
                for file in os.listdir(result_dir):
                    file_path = os.path.join(result_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
            else:
                # 폴더가 아예 없으면 생성
                os.makedirs(result_dir, exist_ok=True)
                    
            for file in target_files:
                src_file = os.path.join(output_dir, file)
                dst_file = os.path.join(result_dir, file)
                
                if os.path.isfile(src_file):
                    shutil.move(src_file, dst_file)
                    logger.info(f"Moved: {src_file} -> {dst_file}")
            
            logger.info(f"[{task_id}] Done: {result_dir}")
            return result_dir
        
        logger.info(f"[{task_id}] Error: failed to make {output_dir}")
        return None

    except Exception as e:
        logger.info(f"[{task_id}] Error: {e}")
        return None






######## 2d to 3d image.yaml(zero123-xl) ########
async def run_dreamgaussian2d(image_path: str, task_id: str, elevation: int = 0) -> Optional[str]:
    try:
        task_progress[task_id] = "processing"
        logger.info(f"task_id: {task_id}, function: run_dreamgaussian_2d image.yaml")

        # png가 아니면 png로 바꿔주기
        image_path = await convert_to_png(task_id, image_path)
        if image_path is None:
            return None

        # 파일명 설정
        name, ext = os.path.splitext(os.path.basename(image_path)) 
        # name은 오직 파일 명
        # 입력: "/app/backend-dreamgaussian/uploads/luigi.png"
        # 출력: "luigi.png"
        processed_image = f"{name}_rgba.png"
        input_image_path = os.path.join(DREAMGAUSSIAN_DIR, "data", processed_image)
        output_dir = os.path.join(DREAMGAUSSIAN_DIR, "logs", "outputs")
        result_dir = os.path.join(RESULT_DIR, f"{task_id}")

        # 1. image preprocessing (process.py)
        process_command = f"python3 process.py {image_path}"
        process = await asyncio.create_subprocess_exec(
            *shlex.split(process_command),
            cwd=os.path.join(DREAMGAUSSIAN_DIR),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            logger.info(f"[{task_id}] process.py failed:\n{stderr.decode().strip()}")
            return None
        
        # image move (UPLOAD_DIR -> data/)
        if os.path.exists(input_image_path):
            os.remove(input_image_path)
        shutil.move(image_path, input_image_path)

        # stage 1 (main.py)
        command_1 = f"""
        python3 main.py \
          --config configs/image.yaml  \
          input=data/{processed_image} \
          save_path=outputs/{name}_mesh \
          elevation={elevation} \
          force_cuda_rast=True
        """

        # stage 2 (main2.py)
        command_2 = f"""
        python3 main2.py \
          --config configs/image.yaml  \
          input=data/{processed_image} \
          save_path=outputs/{name}_mesh \
          elevation={elevation} \
          force_cuda_rast=True
        """
        
        # 프로세스 실행
        for i, command in enumerate([command_1, command_2]):
            process = await asyncio.create_subprocess_exec(
                *shlex.split(command),
                cwd=DREAMGAUSSIAN_DIR,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                logger.info(f"[{task_id}] {line.decode().strip()}")

                
            await process.wait()
            if process.returncode != 0:
                logger.info(f"[{task_id}] Error: {command} failed with code {process.returncode}")
                return None
            
        
        task_progress[task_id] = "processing (100%)"   

        if os.path.exists(output_dir):
            target_files = [
                f"{name}_mesh.obj",
                f"{name}_mesh.mtl",
                f"{name}_mesh_albedo.png"
            ]

            if os.path.exists(result_dir):
                # 폴더 내부의 파일만 삭제
                for file in os.listdir(result_dir):
                    file_path = os.path.join(result_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
            else:
                # 폴더가 아예 없으면 생성
                os.makedirs(result_dir, exist_ok=True)
                    
            for file in target_files:
                src_file = os.path.join(output_dir, file)
                dst_file = os.path.join(result_dir, file)
                
                if os.path.isfile(src_file):
                    shutil.move(src_file, dst_file)
                    logger.info(f"Moved: {src_file} -> {dst_file}")
            
            dst_file = os.path.join(result_dir, target_files[2])
            await upscale_image(task_id, dst_file)

            logger.info(f"[{task_id}] Done: {result_dir}")
            return result_dir
        
        logger.info(f"[{task_id}] Error: failed to make {output_dir}")
        return None

    except Exception as e:
        logger.info(f"[{task_id}] Error: {e}")
        return None




######## 2d to 3d image_sai.yaml(stable-zero123) ########
async def run_dreamgaussian2d2(image_path: str, task_id: str, elevation: int = 0) -> Optional[str]:
    try:
        task_progress[task_id] = "processing"
        logger.info(f"task_id: {task_id}, function: run_dreamgaussian_2d image_sai.yaml")


        # png가 아니면 png로 바꿔주기
        image_path = await convert_to_png(task_id, image_path)
        if image_path is None:
            return None

        # 파일명 설정
        name, ext = os.path.splitext(os.path.basename(image_path)) 
        # name은 오직 파일 명
        # 입력: "/app/backend-dreamgaussian/uploads/luigi.png"
        # 출력: "luigi.png"
        processed_image = f"{name}_rgba.png"
        input_image_path = os.path.join(DREAMGAUSSIAN_DIR, "data", processed_image)
        output_dir = os.path.join(DREAMGAUSSIAN_DIR, "logs", "outputs")
        result_dir = os.path.join(RESULT_DIR, f"{task_id}")

        # 1. image preprocessing (process.py)
        process_command = f"python3 process.py {image_path}"
        process = await asyncio.create_subprocess_exec(
            *shlex.split(process_command),
            cwd=os.path.join(DREAMGAUSSIAN_DIR),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            logger.info(f"[{task_id}] process.py failed:\n{stderr.decode().strip()}")
            return None
        
        # image move (UPLOAD_DIR -> data/)
        if os.path.exists(input_image_path):
            os.remove(input_image_path)
        shutil.move(image_path, input_image_path)

        # stage 1 (main.py)
        command_1 = f"""
        python3 main.py \
        --config configs/image_sai.yaml \
        input=data/{processed_image} \
        save_path=outputs/{name}_mesh \
        elevation={elevation} \
        zero123.use_elevation_cond=false \
        zero123.elevation_range=[0,0] \
        zero123.elevation_noise_std=0 \
        zero123.use_azimuth_cond=false \
        zero123.azimuth_range=[0,0] \
        zero123.azimuth_noise_std=0 \
        force_cuda_rast=True
        """

        # stage 2 (main2.py)
        command_2 = f"""
        python3 main2.py \
        --config configs/image_sai.yaml \
        input=data/{processed_image} \
        save_path=outputs/{name}_mesh \
        elevation={elevation} \
        zero123.use_elevation_cond=false \
        zero123.elevation_range=[0,0] \
        zero123.elevation_noise_std=0 \
        zero123.use_azimuth_cond=false \
        zero123.azimuth_range=[0,0] \
        zero123.azimuth_noise_std=0 \
        force_cuda_rast=True
        """
        
        # 프로세스 실행
        for i, command in enumerate([command_1, command_2]):
            process = await asyncio.create_subprocess_exec(
                *shlex.split(command),
                cwd=DREAMGAUSSIAN_DIR,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                logger.info(f"[{task_id}] {line.decode().strip()}")

                
            await process.wait()
            if process.returncode != 0:
                logger.info(f"[{task_id}] Error: {command} failed with code {process.returncode}")
                return None
            
        
        task_progress[task_id] = "processing (100%)"   

        if os.path.exists(output_dir):
            target_files = [
                f"{name}_mesh.obj",
                f"{name}_mesh.mtl",
                f"{name}_mesh_albedo.png"
            ]

            if os.path.exists(result_dir):
                # 폴더 내부의 파일만 삭제
                for file in os.listdir(result_dir):
                    file_path = os.path.join(result_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
            else:
                # 폴더가 아예 없으면 생성
                os.makedirs(result_dir, exist_ok=True)
                    
            for file in target_files:
                src_file = os.path.join(output_dir, file)
                dst_file = os.path.join(result_dir, file)
                
                if os.path.isfile(src_file):
                    shutil.move(src_file, dst_file)
                    logger.info(f"Moved: {src_file} -> {dst_file}")
            
            dst_file = os.path.join(result_dir, target_files[2])
            await upscale_image(task_id, dst_file)

            logger.info(f"[{task_id}] Done: {result_dir}")
            return result_dir
        
        logger.info(f"[{task_id}] Error: failed to make {output_dir}")
        return None

    except Exception as e:
        logger.info(f"[{task_id}] Error: {e}")
        return None




######## worker ########


@asynccontextmanager
async def lifespan(app: FastAPI):
    async def worker():
        while True:
            task: TaskItem = await task_queue.get()
            task_progress[task.id] = "processing"
            try:
                if task.type == TaskType.TEXT_TO_3D:
                    result_path = await run_dreamgaussian_text_image(task.id, task.data)
                    if result_path:
                        task_result_paths[task.id] = result_path
                        task_progress[task.id] = "done"
                        logger.info(f"[DEBUG] output test: {result_path}")
                    else:
                        task_progress[task.id] = "error: model generation failed"

                elif task.type == TaskType.IMAGE_TO_3D:
                    image_path = task.data["image_path"]
                    result_path = await run_dreamgaussian2d(image_path, task.id)
                    if result_path:
                        task_result_paths[task.id] = result_path
                        task_progress[task.id] = "done"
                        logger.info(f"[DEBUG] output test: {result_path}")
                    else:
                        task_progress[task.id] = "error: model generation failed"
                
                # 테스트 모드 (비동기 작업 시뮬레이션)
                elif task.type in (TaskType.TEXT_TO_3D_TEST, TaskType.IMAGE_TO_3D_TEST):
                    result_path = await handle_test(task)
                    if result_path:
                        task_result_paths[task.id] = result_path
                        task_progress[task.id] = "done"
                        logger.info(f"[DEBUG] test output test: {result_path}")
                    else:
                        task_progress[task.id] = "error: model generation failed"
                
            except Exception as e:
                task_progress[task.id] = f"error: {str(e)}"

    # Run wordker
    logger.info("Server start.")
    asyncio.create_task(worker())

    yield
    
    # Server end
    logger.info("Server end.")

######## fastapi ########

app = FastAPI(lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://my-character.cho0h5.org",
        "http://my-little-precious-character.iptime.org:3000"
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
    while task_id in task_progress:
        task_id = uuid4().hex

    task_type = TaskType.TEXT_TO_3D if mode == "prod" else TaskType.TEXT_TO_3D_TEST
    task = TaskItem(id=task_id, type=task_type, data={"prompt": prompt})
    logger.info(f"[DEBUG] task_type: {task_type} {mode}, task_id: {task_id}, prompt: {prompt}")

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
    while task_id in task_progress:
        task_id = uuid4().hex

    file_extension = os.path.splitext(file.filename)[1]
    file_path = os.path.join(UPLOAD_DIR, f"{task_id}{file_extension}")

    # Store file
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    task_type = TaskType.IMAGE_TO_3D if mode == "prod" else TaskType.IMAGE_TO_3D_TEST
    task = TaskItem(id=task_id, type=task_type, data={"image_path": file_path})
    logger.info(f"[DEBUG] task_type: {task_type} {mode}, task_id: {task_id}")

    await task_queue.put(task)
    task_progress[task_id] = "queued"

    # Response
    return {"task_id": task_id}

@app.websocket("/text-to-3d/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info(f"[DEBUG] /text-to-3d/ws, connected host: {websocket.client.host}:{websocket.client.port}")
    try:
        task_id = await websocket.receive_text()
        start_time = time.time()
        while True:
            await asyncio.sleep(1)
            status = task_progress.get(task_id, "unknown")
            
            if status == "processing":
                status = f"processing ({min(99, int((time.time() - start_time) * 0.5))}%)"
                
            logger.info(f"websocket: task_id: {task_id}, status: {status}")
            await websocket.send_text(f"status: {status}")
            
            if status == "done" or status.startswith("error"):
                break

    finally:
        await websocket.close()

@app.websocket("/image-to-3d/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info(f"[DEBUG] /image-to-3d/ws, connected host: {websocket.client.host}:{websocket.client.port}")
    try:
        task_id = await websocket.receive_text()
        start_time = time.time()
        while True:
            await asyncio.sleep(1)
            status = task_progress.get(task_id, "unknown")
            
            if status == "processing":
                status = f"processing ({min(99, int((time.time() - start_time) * 1.2))}%)"

            logger.info(f"websocket: task_id: {task_id}, status: {status}")
            await websocket.send_text(f"status: {status}")
            
            if status == "done" or status.startswith("error"):
                break


    finally:
        await websocket.close()

@app.get("/text-to-3d")
@app.get("/image-to-3d")
async def get_result(task_id: str,  type: FileType = Query(FileType.obj)):
    try:
        # 대소문자 무시하고 Enum으로 변환
        file_type = FileType(type.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid file type. Allowed values: obj, mtl, albedo")
    
    
    status = task_progress.get(task_id)
    logger.info(f"[DEBUG] task_id: {task_id}, file type: {type}, status: {status}")
    path = task_result_paths.get(task_id)


    # 문제 생긴 경우. 나중에 raise말고 return으로 바꾸기.
    if status is None:
        raise HTTPException(status_code=404, detail="Task not found")
    elif status.startswith("error:"):
        raise HTTPException(status_code=400, detail=status[7:])
    elif status != "done":
        raise HTTPException(status_code=400, detail="Task not complete")
    
    if path is None:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if type == FileType.obj:
        filename = f"{task_id}_mesh.obj"
    elif type == FileType.mtl:
        filename = f"{task_id}_mesh.mtl"
    elif type == FileType.albedo:
        filename = f"{task_id}_mesh_albedo.png"

    path = os.path.join(path, filename)
    if not path or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="File not found")

    # # task progress, task_result_path에서 task_id 제거
    # task_progress.pop(task_id, None)
    # task_result_paths.pop(task_id, None)
    
    return FileResponse(path, media_type="application/octet-stream", filename=filename)
