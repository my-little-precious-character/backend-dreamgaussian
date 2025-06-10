# #!/usr/bin/env python3
# # pure_sd.py
# # 입력 이미지 없이 프롬프트만 받아 Stable Diffusion으로 그림을 그리는 기본 코드

# import argparse
# from diffusers import StableDiffusionPipeline
# import torch
# from PIL import Image

# def main():
#     parser = argparse.ArgumentParser(description="Stable Diffusion 1.5로 텍스트 프롬프트 기반 이미지 생성")
#     parser.add_argument("--prompt", type=str, required=True, help="생성할 이미지의 프롬프트(예: 'a cute mystical creature, digital art')")
#     parser.add_argument("--negative", type=str, default="", help="원하지 않는 요소 (네거티브 프롬프트)")
#     parser.add_argument("--output", type=str, default="result.png", help="저장할 파일명")
#     parser.add_argument("--height", type=int, default=512, help="이미지 높이(px)")
#     parser.add_argument("--width", type=int, default=512, help="이미지 폭(px)")
#     parser.add_argument("--steps", type=int, default=30, help="샘플링 스텝(높을수록 품질↑)")
#     parser.add_argument("--cfg_scale", type=float, default=7.5, help="가이던스(조건 강도)")
#     args = parser.parse_args()

#     # GPU 지원 시 float16, CPU는 float32
#     torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     # Stable Diffusion 1.5 파이프라인 로드
#     pipe = StableDiffusionPipeline.from_pretrained(
#         "runwayml/stable-diffusion-v1-5", torch_dtype=torch_dtype
#     ).to(device)


#     # (옵션) VRAM 절약이 필요하면 아래 두 줄 추가
#     # pipe.enable_attention_slicing()
#     # pipe.enable_model_cpu_offload()

#     # 이미지 생성
#     result = pipe(
#         prompt=args.prompt,
#         negative_prompt=args.negative if args.negative else None,
#         height=args.height,
#         width=args.width,
#         num_inference_steps=args.steps,
#         guidance_scale=args.cfg_scale
#     ).images[0]

#     result.save(args.output)
#     print(f"✅ 이미지가 저장되었습니다: {args.output}")

# if __name__ == "__main__":
#     main()




#############

# test.py --prompt "a cute one woman, T-pose, full-shot, full-body, no-background" --negative "low quality, worst quality" --output creature.png 

#!/usr/bin/env python3
# sdxl_text2img.py
import argparse
from diffusers import StableDiffusionXLPipeline
import torch

def main():
    parser = argparse.ArgumentParser(description="Stable Diffusion XL(SDXL) 텍스트 프롬프트 기반 이미지 생성")
    parser.add_argument("--prompt", type=str, required=True, help="생성할 이미지의 프롬프트")
    parser.add_argument("--negative", type=str, default="", help="네거티브 프롬프트")
    parser.add_argument("--output", type=str, default="result.png", help="저장할 파일명")
    parser.add_argument("--height", type=int, default=1024, help="이미지 높이(px)")
    parser.add_argument("--width", type=int, default=1024, help="이미지 폭(px)")
    parser.add_argument("--steps", type=int, default=30, help="샘플링 스텝")
    parser.add_argument("--cfg_scale", type=float, default=7.5, help="가이던스(조건 강도)")
    args = parser.parse_args()

    # SDXL 모델은 1024x1024 권장, VRAM 12GB↑ 권장
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch_dtype
    ).to(device)

    # VRAM 부족시 활성화
    # pipe.enable_attention_slicing()
    # pipe.enable_model_cpu_offload()

    image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative if args.negative else None,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg_scale
    ).images[0]

    image.save(args.output)
    print(f"✅ SDXL 이미지가 저장되었습니다: {args.output}")

if __name__ == "__main__":
    main()
