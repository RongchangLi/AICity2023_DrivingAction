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
