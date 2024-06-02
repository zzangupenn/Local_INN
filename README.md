# Local_INN: Localization with Invertible Neural Network

## Introduction


https://github.com/zzangupenn/Local_INN/assets/43392543/9790c0c2-55c8-4c64-88d1-de944bc3ebab




## Installization

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
