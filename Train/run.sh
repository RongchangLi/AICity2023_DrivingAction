#CUDA_VISIBLE_DEVICES=2,3 bash ./exp/aicity3/aicity3_b16_f8_r4_224/run.sh
#CUDA_VISIBLE_DEVICES=2,3 bash ./exp/aicity3/aicity3_b16_f8_r8_224/run.sh
#CUDA_VISIBLE_DEVICES=2,3 bash ./exp/aicity3/aicity3_b16_f8_r16_224/run.sh
#CUDA_VISIBLE_DEVICES=2,3 bash ./exp/aicity3/aicity3_k710_b16_f8_r4_224/run.sh
CUDA_VISIBLE_DEVICES=0,1 bash ./exp/aicity3/aicity3_k710_b16_f8_r8_224/run.sh
CUDA_VISIBLE_DEVICES=0,1 bash ./exp/aicity3/aicity3_k710_b16_f8_r16_224/run.sh
#CUDA_VISIBLE_DEVICES=2,3 bash ./exp/aicity3/frozen_aicity3_b16_f8_r4_224/run.sh
#CUDA_VISIBLE_DEVICES=2,3 bash ./exp/aicity3/frozen_aicity3_b16_f8_r8_224/run.sh
#CUDA_VISIBLE_DEVICES=2,3 bash ./exp/aicity3/frozen_aicity3_b16_f8_r16_224/run.sh
#CUDA_VISIBLE_DEVICES=2,3 bash ./exp/aicity3/frozen_aicity3_k710_b16_f8_r4_224/run.sh
CUDA_VISIBLE_DEVICES=0,1 bash ./exp/aicity3/frozen_aicity3_k710_b16_f8_r8_224/run.sh
CUDA_VISIBLE_DEVICES=0,1 bash ./exp/aicity3/frozen_aicity3_k710_b16_f8_r16_224/run.sh

CUDA_VISIBLE_DEVICES=0,1 bash ./exp/aicity3/aicity3_k710_l14_f8_r4_336/run.sh
