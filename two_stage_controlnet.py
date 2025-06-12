# two_stage_controlnet.py
#Stable Diffusion으로 베이스 이미지 생성
#생성된 이미지를 ControlNet 파이프라인에 입력

# 명령어 예시
# python3 two_stage_controlnet.py --prompt "a cute mystical creature, digital art" --negative "low quality, worst quality" --output creature.png
# python3 two_stage_controlnet.py --control_image "./new_image.png" --prompt "a cute one woman, T-pose, full-shot, full-body, no-background" --negative "low quality, worst quality, colorful_background" --output creature1.png
#  python3 two_stage_controlnet.py --control_image "./new_image.png" --prompt "a single young woman, t-pose, full body, arms outstretched, facing forward, standing, no background, photorealistic, studio lighting, centered" --negative "low quality, worst quality, multiple people, two faces, extra limbs, extra arms, extra legs, mutated hands, mutated legs, deformed body, merged faces, merged bodies, back_head, shadow, floor, cropped, duplicate, strange, distorted" --output creature1.png

# prompt "a single young woman, t-pose, full body, arms outstretched, facing forward, standing, no background, photorealistic, studio lighting, centered"
# negative "low quality, worst quality, multiple people, two faces, extra limbs, extra arms, extra legs, mutated hands, mutated legs, deformed body, merged faces, merged bodies, back_head, shadow, floor, cropped, duplicate, strange, distorted"

import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image
import argparse

# CLI 인자 파싱
parser = argparse.ArgumentParser(description="Stable Diffusion 2.1 + ControlNet image generator")
parser.add_argument("--prompt", type=str, required=True, help="텍스트 프롬프트 (이미지 내용 묘사)")
parser.add_argument("--negative", type=str, default="", help="부정 프롬프트 (원하지 않는 이미지 요소)")
parser.add_argument("--output", type=str, default="result.png", help="저장할 출력 이미지 경로")
parser.add_argument("--control_image", type=str, default=None, help="ControlNet 용 입력 이미지 경로 (에지 지도 등)")
parser.add_argument("--height", type=int, default=1024, help="출력 이미지 높이(px)")
parser.add_argument("--width", type=int, default=1024, help="출력 이미지 폭(px)")
parser.add_argument("--steps", type=int, default=35, help="확산 단계 수 (클수록 이미지 품질 향상)")
parser.add_argument("--cfg_scale", type=float, default=7.5, help="텍스트 조건 CFG 강도(guidance_scale)")
parser.add_argument("--control_scale", type=float, default=0.7, help="ControlNet 적용 강도(controlnet_conditioning_scale)")
parser.add_argument("--controlnet_model", type=str, default="lllyasviel/sd-controlnet-openpose", help="ControlNet 모델 ID")
args = parser.parse_args()

# 1. 모델 ID 설정 및 모델 로드
base_model_id = "runwayml/stable-diffusion-v1-5"
# controlnet_model_id = "lllyasviel/sd-controlnet-canny"
controlnet_model_id = args.controlnet_model

# Diffusers 모델 다운로드 및 로드 (자동으로 권한 인증 필요 시 로그인)
controlnet = ControlNetModel.from_pretrained(controlnet_model_id, torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_id, controlnet=controlnet, torch_dtype=torch.float16,
    safety_checker=None, feature_extractor=None, requires_safety_checker=False
)

# 2. 추론 장치 및 메모리 설정
# GPU 사용 설정 (GPU 없으면 .to("cpu")로 변경하거나 torch_dtype=float32로 재로드)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)  # 더 빠른 스케줄러로 교체
pipe.enable_attention_slicing()       # 메모리 절약 모드 활성화
pipe.enable_model_cpu_offload()       # CPU-Offload 활성화 (VRAM 부족 시 안정적, 충분하면 주석 처리)

# 3. ControlNet 입력 이미지 준비 (없으면 검은 빈 이미지 사용)
if args.control_image:
    control_image = Image.open(args.control_image).convert("RGB")
    # ※ Canny 모델의 경우, control_image는 흰색 에지(선)만 있는 흑백 이미지여야 합니다.
    #    필요한 경우 여기서 OpenCV 등을 통해 args.control_image에 Canny 필터를 적용하세요.
else:
    control_image = Image.new("RGB", (args.width, args.height), color=(0, 0, 0))

# 4. 이미지 생성
result = pipe(
    args.prompt,
    negative_prompt=args.negative if args.negative else None,
    image=control_image,
    controlnet_conditioning_scale=args.control_scale,
    num_inference_steps=args.steps,
    guidance_scale=args.cfg_scale,
    height=args.height, width=args.width
).images[0]

# 5. 결과 저장
result.save(args.output)
print(f"Saved result image to {args.output}")
