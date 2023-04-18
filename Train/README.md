

This repo is the solution for 7th AICITY (2023) Challenge Track 3 - Naturalistic Driving Action Recognition.
![framework](framework.png)

## Running environment

Please refer to the [UniFormerv2](https://github.com/OpenGVLab/UniFormerV2) to install pytorch and pyslow.

## Preprocessing
TODO

## Train
It needs 2 RTX3090 GPUS (totally 48G ) to perform the training experiments.
First run this command to go into the training folder. 
```bash
cd Train
```
In the [exp/aicity3](Train/exp/aicity3) folder, there are folders of experiments of different setting.
To train a model, you should first specific the DATA.PATH_TO_DATA_DIR in the run.sh as your own path. Then please run:
```
CUDA_VISIBLE_DEVICES=[your_GPU_indexes] python exp/aicity3/[EXPERIMENTS_FOLDER]/run.sh
```
The training results are in the  _experiments_fold_, shown as follows:
>   * EXPERIMENTS_FOLDER
>     * checkpoint_[VIEW]_[TRAIN_MANNER]_zero
>       * checkpoints
>         * checkpoint_epoch_000**.pyth
>         * ...
>       * stdout.log
>       * ...
>     * run.sh
>      * config.yaml

We use the model weights of the last epoch to predict the snippet-level action probabilities.


## Inference
The format of inference should be similar with the A2 dataset, which is provided by 2023 AI City Challenge. The format of A2 dataset as follows:
>   * A2
>     * user_id_*
>       * CAMERAVIEW_user_id_*.MP4
>       * CAMERAVIEW_user_id_*.MP4
>       * CAMERAVIEW_user_id_*.MP4
>       * ...
>     * video_ids.csv

### Reproduce
The checkpoints after trainning process can be downloaded [here](https://baidu.com), which includes all the checkpoints of different trainning datatypes, camera views and sampling strategy. After downloading all the checkpoints, please put all files into ./X3D_inference/checkpoint_submit/
```bash
cd Inference
```
The inference has two stages, including action probability calibration result generation and efficient action localization. Please run the following commands to reproduce our results in sequence.
```bash
python generation_probability.py --cfg configs/Kinetics/X3D_L.yaml NUM_GPUS 1 TRAIN.ENABLE False DATA.PATH_TO_DATA_DIR A2
```
The results of the first stage will appear in ./probability_results

```bash
python inference_finalresult.py --cfg configs/Kinetics/X3D_L.yaml NUM_GPUS 1 TRAIN.ENABLE False DATA.PATH_TO_DATA_DIR A2
```
DATA.PATH_TO_DATA_DIR: path to Test Dataset (e.g., A2, B)
Submission file appeare in ./output

#### Quickly reproduce A2 results
If you want to reproduce it quickly, the npy files for the first stage can be downloaded from [here](https://baidu.com). After downloading, please put these files in ./probability_results. Then run the following command.

```bash
python inference_finalresult.py --cfg configs/Kinetics/X3D_L.yaml NUM_GPUS 1 TRAIN.ENABLE False DATA.PATH_TO_DATA_DIR A2
```



## Acknowledgement

This repository is built based on [UniFormerv2](https://github.com/OpenGVLab/UniFormerV2) and [Winner of 2022](https://github.com/VTCC-uTVM) repository.


