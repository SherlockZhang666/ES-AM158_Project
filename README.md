# ES-AM158_Project
RL-Tuned Gain Scheduling for Robotic Arm Reaching Task

# Set up (linux)
```
conda create -n rlpj  python==3.10 -y
conda activate rlpj
```

## Install corresponding torch GPU version (eg: on cuda 12.4)
```
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
```

## Other requirements
```
pip install numpy "gymnasium[classic-control]" stable-baselines3 matplotlib
```

## Visualization without GUI (optional)
```
pip install imageio-ffmpeg
```