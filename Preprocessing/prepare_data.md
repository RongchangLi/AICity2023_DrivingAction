# Data Preparation For Training
Download the training data (A1), which is provided by 2023 ACity Challenge and put the download file into ./X3D_training/data/A1/

Please note that due to the obvious missing data in 'user_id_86356.csv', we need to add the 'Dashboard_User_id_86356_5', 'Rear_view_user_id_86356_5 ' and 'Right_side_window_user_id_86356_5' in which we decided to populate the label in their corresponding positions with 'Class 1' and change the 'End Timed' in their next line from '0:03:32' to '0:08:32'.

```bash
cd X3D_training
```
Splitting training data into multiple video segments using the following command (around 04 hours of time consuming, the splitted data can be download [here](www.baidu.com)(for accessable person only)):
```bash
python cut_video.py
```
Then, generate the train/val/test csv files by:
```bash
python create_csv.py
```

After executing the above command, the output data is located in ./X3D_training/data/A1_cut_video/ and the output tarin/val/test.csv is located in ./X3D_training/data/A1_cut_video/train_val_test_csv/
 
The Dataset is then splitted into video segments and put into different folder of labels based on ground truth (user_id_*.csv). The splitted files is formated as follows:

>   * data
>     * 0
>       * VIDEO1.MP4
>       * VIDEO2.MP4
>       * VIDEO3.MP4
>       * ...
>       ...
>     * 15
>       * VIDEO1.MP4
>       * VIDEO2.MP4
>       * VIDEO3.MP4
