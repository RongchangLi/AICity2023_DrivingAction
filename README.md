

This repo is the solution for 7th [AICITY](https://www.aicitychallenge.org/2023-challenge-tracks/) (2023) Challenge Track 3 - Naturalistic Driving Action Recognition.
![framework](framework.png)

## Running environment

Please refer to the [UniFormerv2](https://github.com/OpenGVLab/UniFormerV2) to prepare the running environment.

## Preprocessing
Please run command below to go into the preprocessing folder:
```bash
cd Preprocessing
```

<!-- ./X3D_training/data/A1/ -->
<!-- Please note that due to the obvious missing data in 'user_id_86356.csv', we need to add the 'Dashboard_User_id_86356_5', 'Rear_view_user_id_86356_5 ' and 'Right_side_window_user_id_86356_5' in which we decided to populate the label in their corresponding positions with 'Class 1' and change the 'End Timed' in their next line from '0:03:32' to '0:08:32'.
 -->
First, download the training data and annotation files (A1), which are provided by 2023 AICity Challenge. Then put the downloaded data into **YOUR_DATA_PATH**.
As there exists some errors in the official annotaion files 'user_id_86356.csv', we recommand to replace it by our modified files saved in  [Preprocessing/A1_new_annotation](Preprocessing/A1_new_annotation).
Next modify the __data_homepath__ parameter in [Preprocessing/cut_video.py](Preprocessing/cut_video.py#L31) and [Preprocessing/create_csv.py](Preprocessing/create_csv.py#L6) as your own data path.

After finishing the preparations, run commands below to cut original long training videos into meaningful segments (It needs about 04 hours.) <!-- the splitted data can be download [here](www.baidu.com)(for accessable person only)) -->:
```bash
python cut_video.py
```
Then, generate the training files (*.csv) by running:
```bash
python create_csv.py
```
<!-- After executing the above command, the cutted videos is saved in ./X3D_training/data/A1_cut_video/ and the output tarin/val/test.csv is located in ./X3D_training/data/A1_cut_video/train_val_test_csv/ -->
Finally, the total structure of YOUR_DATA_PATH should be same as follows:
The splitted files is formated as follows:
>   * YOUR_DATA_PATH
>     * A1
>       * user_id_*
>         * CAMERAVIEW_user_id_*.MP4
>         * user_id_*.csv
>       * ...
>     * A1_cut_video
>       * new_spllit_by_id_A1_total
>         * expand_zero
>           * train_dashboard.csv
>           * val_dashboard.csv
>           * test_dashboard.csv
>           * ...
>         * original_zero
>       * 0
>         * *.MP4 
>       * 1
>       * ...

## Train
It needs 2 RTX3090 GPUS ( totally 48G ) to perform the training experiments.

First run this command to go into the training folder. 
```bash
cd Train
```
In the [Train/exp/aicity3](Train/exp/aicity3) folder, there experiment folders to run all of our used models.
To train a model, you should first specific the **DATA.PATH_TO_DATA_DIR** in the **run.sh** files as your own path (should be YOUR_DATA_PATH/A1_cut_video).

Then please run:
<!--The structure of PATH_TO_DATA_DIR should be like (It actually is the ):
>   * PATH_TO_DATA_DIR
>     * new_spllit_by_id_A1_total
>       * expand_zero
>         * train_dashboard.csv
>         * val_dashboard.csv
>         * test_dashboard.csv
>         * ...
>       * original_zero
>     * 0
>       * *.MP4 
>     * 1
>     * ...
-->
```
CUDA_VISIBLE_DEVICES=[your_GPU_indexes] python exp/aicity3/[EXPERIMENTS_FOLDER]/run.sh
```
The training results are in the  **EXPERIMENTS_FOLDER**, the structure of which are shown as follows:
>   * EXPERIMENTS_FOLDER
>     * checkpoint_[VIEW]_[TRAIN_MANNER]_zero
>       * checkpoints
>         * checkpoint_epoch_000**.pyth
>         * ...
>       * stdout.log
>       * ...
>     * run.sh
>     * config.yaml

We use the weights of the last epoch to predict the snippet-level action probabilities.


## Inference
### General folder structure for Inference
<!-- The format of inference should be similar with the A2 dataset, which is provided by 2023 AI City Challenge. The format of A2 dataset as follows: -->
For custom dataset, please put the videos to be tested in the YOUR_TEST_DATA folder. And organize the YOUR_TEST_DATA folder as follows:
>   * YOUR_TEST_DATA
>     * user_id_*
>       * CAMERAVIEW_user_id_*.MP4
>       * CAMERAVIEW_user_id_*.MP4
>       * CAMERAVIEW_user_id_*.MP4
>       * ...
>     * video_ids.csv
### Reproduce
To reproduce our results, you should run all the experiments in the [Train/exp/aicity3](Train/exp/aicity3) folder. We recommend to directly download our pretrianed checkpoints [here](https://drive.google.com/drive/folders/1ZqcT_Z3rqEXrTSe3k_WpYpmhHBPAgnCF?usp=share_link). It includes all the checkpoints of different trainning datatypes, camera views and sampling strategy.
After training or downloading all the checkpoints, please put them into the [Inference/checkpoint_submit](Inference/checkpoint_submit) folder.
Then organize the A2 dataset provided by 2023 AI City Challenge as mentioned manner above.

After finishing preparation, please run command below to go into the inference folder:
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
The generated results are in the [Inference/probability_results](Inference/probability_results) folder. Then run command below to generate the final submission files.
```bash
python inference_finalresult.py --cfg configs/Kinetics/X3D_L.yaml NUM_GPUS 1 TRAIN.ENABLE False DATA.PATH_TO_DATA_DIR [YOUR_TEST_DATA]
```
<!-- DATA.PATH_TO_DATA_DIR: path to Test Dataset (e.g., A2, B) -->
The generated submission files are saved in [Inference/output](Inference/output) folder.

#### Quickly reproduce our results on public leaderboard
If you want to reproduce it quickly, the .npy files for the first stage can be downloaded from [here](https://drive.google.com/drive/folders/1ZqcT_Z3rqEXrTSe3k_WpYpmhHBPAgnCF?usp=share_link). After downloading, please put these files in ./probability_results. Then run the following command.

```bash
python inference_finalresult.py --cfg configs/Kinetics/X3D_L.yaml NUM_GPUS 1 TRAIN.ENABLE False DATA.PATH_TO_DATA_DIR [YOUR_TEST_DATA]
```


## Acknowledgement

This repository is built based on [UniFormerv2](https://github.com/OpenGVLab/UniFormerV2) and [Winner of 2022](https://github.com/VTCC-uTVM) repository.


