# Docker Setup for GPU (Windows)

This guide provides step-by-step instructions for setting up Docker with GPU support on Windows using WSL.

---

## 1. WSL Version Check

First, verify your WSL installation:

```bash
wsl -l -v
```

### **Case 1:**

```
  NAME                   STATE           VERSION
* docker-desktop         Running         2
  Ubuntu                 Running         2
```

### **Case 2:**

```
  NAME                   STATE           VERSION
* docker-desktop         Running         2
```

If you see **Case 2**, you'll need to install Ubuntu:

```bash
wsl --install -d Ubuntu
```

After installation, **reboot** your machine to ensure the changes take effect. You should now see the output similar to **Case 1**.

---

## 2. WSL Configuration for GPU

Open your WSL instance:

```bash
wsl
```

Then run the following commands to configure GPU support:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit.gpg

sudo sed -i 's|^deb https://|deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit.gpg] https://|' /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install nvidia-container-toolkit
exit
```

---

## 3. Testing GPU in Docker

Verify your Docker GPU setup:

```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-devel-ubuntu22.04 nvidia-smi
```

Expected output (sample):

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.86.05    Driver Version: ...    CUDA Version: 12.1          |
| GPU Name        Persistence-M | Bus-Id | ...                               |
| ... (GPU 정보 출력) ...                                           |
+-----------------------------------------------------------------------------+
```

If you see a similar output, your GPU is correctly configured.

---

## 4. Docker Compose Commands

Build and run your Docker containers:

```bash
docker compose up --build
```

Access the container:

```bash
docker exec -it dreamgaussian bash
```

Restart all running containers:

```bash
docker compose restart
```

Run the main application:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

---
