

This repo is the solution for 7th AICITY (2023) Challenge Track 3 - Naturalistic Driving Action Recognition.
![framework](framework.png)

## Running environment

Please refer to the [UniFormerv2](https://github.com/OpenGVLab/UniFormerV2) to install pytorch and pyslow.

## Preprocessing
TODO

## Train
It needs 2 RTX3090 GPUS ( totally 48G ) to perform the training experiments.
First run this command to go into the training folder. 
```bash
cd Train
```
In the [exp/aicity3](Train/exp/aicity3) folder, there are folders for experiments of different settings.
To train a model, you should first specific the DATA.PATH_TO_DATA_DIR in the run.sh as your own path. And put the generated files to the same path.
The structure of PATH_TO_DATA_DIR should be like:
>   * PATH_TO_DATA_DIR
>     * new_spllit_by_id_A1_20
>       * expand_zero
>         * train_dashboard.csv
>         * val_dashboard.csv
>         * test_dashboard.csv
>         * ...
>       * original_zero
>     * new_spllit_by_id_A1_total
>     * 0
>       * *.MP4 
>     * 1
>     * ...

Then please run:
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

We use the weights of the last epoch to predict the snippet-level action probabilities.


## Inference
<!-- The format of inference should be similar with the A2 dataset, which is provided by 2023 AI City Challenge. The format of A2 dataset as follows: -->
The videos to be inferenced are in the YOUR_TEST_DATA folder. The structure of TEST_DATA folder is shown as follows:
>   * YOUR_TEST_DATA
>     * user_id_*
>       * CAMERAVIEW_user_id_*.MP4
>       * CAMERAVIEW_user_id_*.MP4
>       * CAMERAVIEW_user_id_*.MP4
>       * ...
>     * video_ids.csv
### Reproduce
The pretrianed checkpoints can be downloaded [here](https://baidu.com). It includes all the checkpoints of different trainning datatypes, camera views and sampling strategy. After downloading all the checkpoints, please put them into **./Inference/checkpoint_submit/**
```bash
cd Inference
```

<!-- First, including action probability calibration result generation and efficient action localization. -->
<!-- Please run the following commands to reproduce our results in sequence. -->
There are two steps in the inference process. First, to generate action probabilities by running (please replace YOUR_TEST_DATA with the specific data path):
```bash
python generation_probability.py --cfg configs/Kinetics/X3D_L.yaml NUM_GPUS 1 TRAIN.ENABLE False DATA.PATH_TO_DATA_DIR [YOUR_TEST_DATA]
```
<!-- The results of the first stage will appear in ./probability_results -->
The generated results are in the **./probability_results** folder. Then run command below to generate the final submission files.
```bash
python inference_finalresult.py --cfg configs/Kinetics/X3D_L.yaml NUM_GPUS 1 TRAIN.ENABLE False DATA.PATH_TO_DATA_DIR [YOUR_TEST_DATA]
```
<!-- DATA.PATH_TO_DATA_DIR: path to Test Dataset (e.g., A2, B) -->
The generated submission files are saved in **./output** folder.

#### Quickly reproduce our results on public leaderboard
If you want to reproduce it quickly, the .npy files for the first stage can be downloaded from [here](https://baidu.com). After downloading, please put these files in ./probability_results. Then run the following command.

```bash
python inference_finalresult.py --cfg configs/Kinetics/X3D_L.yaml NUM_GPUS 1 TRAIN.ENABLE False DATA.PATH_TO_DATA_DIR [YOUR_TEST_DATA]
```



## Acknowledgement

This repository is built based on [UniFormerv2](https://github.com/OpenGVLab/UniFormerV2) and [Winner of 2022](https://github.com/VTCC-uTVM) repository.


