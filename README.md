window에서 docker 이용 시 (gpu 사용을 위해)

터미널에 아래 명령어 입력
wsl -l -v

case 1: 
  NAME                   STATE           VERSION
* docker-desktop         Running         2
  Ubuntu                 Running         2

case 2:
  NAME                   STATE           VERSION
* docker-desktop         Running         2

case 2일때:
wsl --install -d Ubuntu

이후 재부팅 => case 1처럼 뜨게 됨.

이후 wsl 설정 (docker 에서 gpu를 사용할 수 있도록)

wsl 

curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit.gpg

sudo sed -i 's|^deb https://|deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit.gpg] https://|' /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install nvidia-container-toolkit
exit


Docker에서 실제로 GPU가 보이는지 테스트

docker run --rm --gpus all nvidia/cuda:12.1.0-devel-ubuntu22.04 nvidia-smi

터미널에 
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.86.05    Driver Version: ...    CUDA Version: 12.1          |
| GPU Name        Persistence-M | Bus-Id | ...
| ... (GPU 정보 출력) ...
+-----------------------------------------------------------------------------+
이런 꼴로 출력된다면 설정 완료.




// docker compose 파일로 build 및 실행

docker compose up --build       # 빌드 후 실시간 실행
<!-- 다른 터미널에서  -->
docker exec -it dreamgaussian bash  # 직접 들어가기


// container들 재시작
docker compose restart


// 실행
uvicorn main:app --host 0.0.0.0 --port 8000