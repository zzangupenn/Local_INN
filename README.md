# Local_INN: Localization with Invertible Neural Network

## Local_INN vs. Particle Filter
Local_INN is red, PF is blue. We can see PF is lagging behind.
![ezgif-6-8b183a4816](https://github.com/zzangupenn/Local_INN/assets/43392543/aca56241-d104-4b9b-beaf-234107e01c4e)

|                               | Local_INN | Particle Filter |
|-------------------------------|-----------|-----------------|
| Output Pose Distribution      | ✅         | ✅               |
| Solving Kidnapping Problem    | ✅         | ✅               |
| Framerate on Jetson Xavier NX | 200Hz+    | 40Hz            |

## Introduction Video

https://github.com/zzangupenn/Local_INN/assets/43392543/9790c0c2-55c8-4c64-88d1-de944bc3ebab

## Installation

1. You can use our Dockerfile build and run the container:
    ```
    git clone --recursive git@github.com:zzangupenn/Local_INN.git
    cd Local_INN
    docker build -t local_inn .
    docker run -ti --rm --gpus all --ipc=host --name local_inn \
    -v [data_dir]:/workspace/data \
    -v ./results:/workspace/results local_inn /bin/bash
    ```

2. Generate data
   ```
   cd levine
   python3 random_sampling.py
   ```
