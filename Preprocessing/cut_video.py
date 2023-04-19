# Import everything needed to edit video clips
from moviepy.editor import *
import pandas as pd
import os
def cut_video(clip1, time_start, time_end, path_video):
    # clip1 = VideoFileClip("test_phone.mp4").subclip(5, 18)
    clip1 = clip1.subclip(time_start, time_end)
    # getting width and height of clip 1
    w1 = clip1.w
    h1 = clip1.h
    
    print("Width x Height of clip 1 : ", end = " ")
    print(str(w1) + " x ", str(h1))
    
    print("---------------------------------------")
    
    # resizing video downsize 50 %
    clip2 = clip1.resize((512, 512))

    # getting width and height of clip 1
    w2 = clip2.w
    h2 = clip2.h
    
    print("Width x Height of clip 2 : ", end = " ")
    print(str(w2) + " x ", str(h2))
    
    print("---------------------------------------")
    clip2.write_videofile(path_video)
    
#create folder & set data path
data_homepath = YOUR_DATA_PATH #Please specific the path to save your data
cut_videopath = data_homepath + '/A1_cut_video'
cut_videoannotation = data_homepath + '/A1_cut_video_annotation'

if not os.path.isdir(cut_videopath):
    os.makedirs(cut_videopath)
else: 
    print("folder already exists.")
for i in range(16):
    data_dir = cut_videopath +'/{}'.format(str(i))
    CHECK_FOLDER = os.path.isdir(data_dir)
    if not CHECK_FOLDER:
        os.makedirs(data_dir)
    else:
        print(data_dir, "folder already exists.")
    print(i)
    
if not os.path.isdir(cut_videoannotation):
    os.makedirs(cut_videoannotation)
else: 
    print("folder already exists.")
    
file_expandzero = open(cut_videoannotation + '/A1_cutvideo_expandzero_total_data.csv','w')
file_originalzero = open(cut_videoannotation + '/A1_cutvideo_originalzero_total_data.csv','w')

for folder_name in os.listdir(data_homepath + '/A1/'):
    # print(folder_name)
    path_folder = data_homepath + '/A1/{}'.format(folder_name)
    path_csv = '{}/{}.csv'.format(path_folder, folder_name)
    # print(path_folder, path_csv)
    if os.path.isdir(path_folder) == True:
        df = pd.read_csv(path_csv)
    else:
        continue

    file_name = ''
    lastfile_name = ''
    newfile_flag = True
    count = 0
    video_list = []
    for i in range(len(df)):
        if  type(df['Filename'][i]) == float or folder_name[8:] not in df['Filename'][i]:
            # df['Filename'][i] = lastfile_name
            file_name = lastfile_name # follow the last file name detected.
        elif folder_name[8:] in df['Filename'][i]:
            # print("df['Filename'][i]", type(df['Filename'][i]), df['Filename'][i])
            file_name = df['Filename'][i].replace('User','user').replace(folder_name, folder_name+'_NoAudio') # rename the filename in csv
            if 'Rear' in file_name:
                file_name = file_name.replace('Rearview','Rear_view')
            lastfile_name = file_name
            newfile_flag = True

            count = 0
        video_list.append([df['Start Time'][i], df['End Time'][i], df['Label (Primary)'][i].strip()[6:]])


        if (i < len(df) - 1 and type(df['Filename'][i+1]) == str and folder_name[8:] in df['Filename'][i+1]) or (i == len(df)-1):
            # print(len(video_list))
            video_list.sort()
            try:
                clip = VideoFileClip("{}/{}.MP4".format(path_folder, file_name))
                ftr = [3600, 60, 1]
                time_start_first = sum([a * b for a, b in zip(ftr, map(int, video_list[0][0].split(':')))])
                # time_end = sum([a * b for a, b in zip(ftr, map(int, video_list[0][1].split(':')))])
                # handle the start cut of normal driving
                if time_start_first != 0 :
                    path_video = cut_videopath + '/{}/{}_{}_{}.MP4'.format(0, file_name, '00:00:00', video_list[0][0])
                    path_video_name2 = cut_videopath + '/{}/{}_{}_0{}.MP4'.format(0, file_name, '00:00:00', video_list[0][0])
                    # if not os.path.isfile(path_video) and not os.path.isfile(path_video_name2):
                    #     cut_video(clip, 0, time_start_first, path_video)

                    if len(video_list[0][0].split(':')[0]) == 2:
                        cut_video(clip, 0, time_start_first, path_video)
                    else:
                        cut_video(clip, 0, time_start_first, path_video_name2)

                    if len(video_list[0][0].split(':')[0]) == 2:
                        file_expandzero.write(path_video + ' ' + str(0) + '\n')
                    else:
                        file_expandzero.write(path_video_name2 + ' ' + str(0) + '\n')
            except Exception as e:
                print(e)
                continue

            for j in range(len(video_list)):
                try:
                    clip = VideoFileClip("{}/{}.MP4".format(path_folder, file_name))
                    ftr = [3600, 60, 1]
                    time_start = sum([a * b for a, b in zip(ftr, map(int, video_list[j][0].split(':')))])
                    time_end = sum([a * b for a, b in zip(ftr, map(int, video_list[j][1].split(':')))])
                    if time_start < time_end:
                        path_video = cut_videopath + '/{}/{}_{}_{}.MP4'.format(
                            video_list[j][2], file_name, video_list[j][0], video_list[j][1])

                        path_video_name2 = cut_videopath + '/{}/{}_0{}_0{}.MP4'.format(
                            video_list[j][2], file_name, video_list[j][0], video_list[j][1])

                        if len(video_list[0][0].split(':')[0]) == 2:
                            cut_video(clip, time_start, time_end, path_video)
                        else:
                            cut_video(clip, time_start, time_end, path_video_name2)

                        if len(video_list[0][0].split(':')[0]) == 2:
                            file_expandzero.write(path_video+' '+video_list[j][2]+'\n')
                            file_originalzero.write(path_video+' '+video_list[j][2]+'\n')
                        else:
                            file_expandzero.write(path_video_name2+' '+video_list[j][2]+'\n')
                            file_originalzero.write(path_video_name2+' '+video_list[j][2]+'\n')

                    if j < len(video_list)-1:
                        # Segment Transition as label 0
                        time_start = sum([a * b for a, b in zip(ftr, map(int, video_list[j][1].split(':')))])
                        time_end = sum([a * b for a, b in zip(ftr, map(int, video_list[j + 1][0].split(':')))])

                        if time_start < time_end:
                            path_video = cut_videopath + '/{}/{}_{}_{}.MP4'.format(0, file_name, video_list[j][1], video_list[j + 1][0])
                            path_video_name2 = cut_videopath + '/{}/{}_0{}_0{}.MP4'.format(0, file_name,video_list[j][1],video_list[j + 1][0])

                            if len(video_list[0][0].split(':')[0]) == 2:
                                cut_video(clip, time_start, time_end, path_video)
                            else:
                                cut_video(clip, time_start, time_end, path_video_name2)

                            if len(video_list[0][0].split(':')[0]) == 2:
                                file_expandzero.write(path_video+' '+str(0)+'\n')
                            else:
                                file_expandzero.write(path_video_name2 + ' ' + str(0) + '\n')

                    count += 1
                except Exception as e:
                    print(e)
                    continue
            video_list = []
            
file_expandzero.close()
file_originalzero.close()



