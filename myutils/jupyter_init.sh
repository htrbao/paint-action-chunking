#!/bin/bash
module load apptainer

# for grete cluster
export VLA_DIR='/mnt/lustre-grete/usr/u12045/vla'
export CODE_DIR='/mnt/lustre-grete/usr/u12045/vla/duci/ReFineVLA_Align'

# # for kiskki cluster
# export VLA_DIR='/projects/extern/kisski/kisski-umg-fairpact-2/dir.project/VLA'
# export CODE_DIR='/projects/extern/kisski/kisski-umg-fairpact-2/dir.project/VLA/duci/ReFineVLA_Align'


IMAGE_PATH=$VLA_DIR/duci/spatial/timage.sif
DATA_DIR=$VLA_DIR
CUDA_DIR=/sw/rev/24.05/rome_ib_cuda_rocky8/linux-rocky8-zen2/gcc-11.4.0/cuda-12.2.1-glm4yc76xa2fewf76gnzelajjscc3iby


# apptainer exec --nv \
#     --fakeroot \
#     --bind $CODE_DIR:/project \
#     --bind $DATA_DIR:/my_data \
#     --bind $CUDA_DIR:$CUDA_DIR \
#     --env HF_HOME=/my_data/hf_home \
#     --env CUDA_HOME=$CUDA_DIR \
#     --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
#     --pwd /project \
#     $IMAGE_PATH \
#         bash -c 'export PYTHONPATH=/project:$PYTHONPATH && jupyter lab --no-browser --ip=0.0.0.0 --port=7001 --allow-root'


apptainer exec --nv \
    --fakeroot \
    --bind $CODE_DIR:/project \
    --bind $DATA_DIR:/my_data \
    --bind $CUDA_DIR:$CUDA_DIR \
    --env HF_HOME=/my_data/hf_home \
    --env CUDA_HOME=$CUDA_DIR \
    --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    --env TRITON_CACHE_DIR=/my_data/cache \
    --env TORCH_EXTENSIONS_DIR=/my_data/cache \
    --env TMPDIR=/my_data/cache \
    --pwd /project \
    $IMAGE_PATH \
        bash -c "
                export PYTHONPATH=/project:$PYTHONPATH && \
                jupyter lab --no-browser --ip=0.0.0.0 --port=7001 --allow-root
                "