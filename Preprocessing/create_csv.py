import os
import csv
import pandas as pd


data_homepath = YOUR_DATA_PATH #Please specific the path to save your data
cut_videopath = data_homepath + '/A1_cut_video'
cut_videoannotation = data_homepath + '/A1_cut_video_annotation'
csv_outpath = cut_videopath + '/new_split_by_id_A1_total'


original_zero_csv = csv_outpath + '/original_zero'
if not os.path.isdir(original_zero_csv):
    os.makedirs(original_zero_csv)
else: 
    print("folder already exists.")

expand_zero_csv = csv_outpath + '/expand_zero'
if not os.path.isdir(expand_zero_csv):
    os.makedirs(expand_zero_csv)
else: 
    print("folder already exists.")

user_id_list = []

# If trained with all data of A1.
for folder_name in os.listdir(data_homepath + '/A1'):
    user_id_list.append(folder_name[8:])
user_id_list.sort()
train_id = [i for i in range(25)]
val_id = [i for i in range(20,25)]
train_user_id_list = str([user_id_list[i] for i in train_id ])
val_user_id_list = str([user_id_list[i] for i in val_id])


k_map = {"dashboard": "Dashboard", "rearview": "Rear", "right": "Right"}
annotation_list = [original_zero_csv, expand_zero_csv]

for annotation in annotation_list:
  for view in ["dashboard", "rearview", "right"]:
      if 'expand' in annotation:   
        f = open(cut_videoannotation + "/A1_cutvideo_expandzero_total_data.csv", "r")
      else:
        f = open(cut_videoannotation + "/A1_cutvideo_originalzero_total_data.csv", "r")
            
      f_train = open(annotation + "/train_{}.csv".format(view), "w")
      f_val = open(annotation + "/val_{}.csv".format(view), "w")
      f_test = open(annotation + "/test_{}.csv".format(view), "w")
  
      for line in f.readlines():
          if line.split('user_id_')[1].split('_NoAudio')[0] in val_user_id_list and k_map[view] in line:
              f_val.write(line)
              f_test.write(line)
          elif line.split('user_id_')[1].split('_NoAudio')[0] in train_user_id_list and k_map[view] in line:
              f_train.write(line)
      f_train.close()
      f_val.close()
  
  for csvtype in ['train','val']:    
    data1 = []
    with open(annotation + '/{}_dashboard.csv'.format(csvtype), newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            data1.append(row)
    
    data2 = []
    with open(annotation + '/{}_rearview.csv'.format(csvtype), newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            data2.append(row)
    
    data3 = []
    with open(annotation +'/{}_right.csv'.format(csvtype), newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            data3.append(row)
    df = pd.concat([pd.DataFrame(data1), pd.DataFrame(data2), pd.DataFrame(data3)])
    df.to_csv(annotation + '/{}_all.csv'.format(csvtype), index=False, header=False)

print("annotation finished")