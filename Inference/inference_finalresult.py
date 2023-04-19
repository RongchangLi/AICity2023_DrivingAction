# !/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math

import numpy
import numpy as np
import os
import sys
import pickle
import torch
import random
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.env import pathmgr
from slowfast.utils.meters import AVAMeter, TestMeter
import cv2
import pandas as pd
import tqdm

"""Wrapper to train and test a video classification model."""
from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args
import time
import numpy as np
from scipy.ndimage import gaussian_filter

logger = logging.get_logger(__name__)
import csv

def imresize(im, dsize):
    '''
    Resize the image to the specified square sizes and
    maintain the original aspect ratio using padding.
    Args:
        im -- input image.
        dsize -- output sizes, can be an integer or a tuple.
    Returns:
        resized image.
    '''
    if type(dsize) is int:
        dsize = (dsize, dsize)
    im_h, im_w, _ = im.shape
    to_w, to_h = dsize
    scale_ratio = min(to_w / im_w, to_h / im_h)
    new_im = cv2.resize(im, (0, 0),
                        fx=scale_ratio, fy=scale_ratio,
                        interpolation=cv2.INTER_AREA)
    new_h, new_w, _ = new_im.shape
    padded_im = np.full((to_h, to_w, 3), 128)
    x1 = (to_w - new_w) // 2
    x2 = x1 + new_w
    y1 = (to_h - new_h) // 2
    y2 = y1 + new_h
    padded_im[y1:y2, x1:x2, :] = new_im
    # print('padd', padded_im)
    return padded_im


class VideoReader(object):
    def __init__(self, source):
        self.source = source
        try:  # OpenCV needs int to read from webcam
            self.source = int(source)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.source)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.source))
        return self

    def __next__(self):
        was_read, frame = self.cap.read()
        if not was_read:
            # raise StopIteration
            ## reiterate the video instead of quiting.
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame = None
            # print('end video')
        return was_read, frame

    def clean(self):
        self.cap.release()
        cv2.destroyAllWindows()


def softmax(x):
    x = np.exp(x - np.expand_dims(np.max(x, axis=1), axis=1))
    x = x / np.expand_dims(x.sum(axis=1), axis=1)
    return x


def generate_gaussian_weights(sigma, length):
    center = length // 2
    x = np.linspace(-center, center, length)
    kernel = np.exp(-x ** 2 / (2 * sigma ** 2))
    return kernel

def generate_gaussian_weights_with_stride(data,  length, stride, sigma = 30):
    weights_list = []
    for i in range(0, len(data), stride):
        sub_data = data[i:i + length]
        if len(sub_data) == length:
            weights = generate_gaussian_weights(sigma, length)
            weights_list.append(weights)
    return np.concatenate(weights_list, axis=0)


@torch.no_grad()
def ensemble_results(cfg, videoids):
    print(videoids)
    total_prob_sq = {}
    video_order = []
    for key, values in videoids.items():
        video_order.append(values)
        video_path = values[1]
        print(video_path)

        overlap_ratio = [0,0.25,0.5,0.75]
        sample_rate1 = [4, 8, 12]
        sample_rate2 = [2, 4, 6]
        frame_nums = [8,16]
        npynum = 0
        use_gussion = True
        views = ['dashboard', 'rearview', 'right']
        datatypes = ['original','expand']
        zeros_pad = [np.zeros((1, 16))]
        npyloader = []
        gussion_weights = []
        dashboard_result = []
        rearview_result = []
        right_result = []
        for view in views:
            for frame_num in frame_nums:
                if frame_num == 16:
                    sample_rate = sample_rate2
                else:
                    sample_rate = sample_rate1
                for datatype in datatypes:
                    for rate in sample_rate:
                        for ratio in overlap_ratio:
                            prob_file = './probability_results/view-{}_frame-{}_rate-{}_datatype-{}_overlapratio-{}.npy'.format(view,frame_num,rate,datatype,ratio)
                            load_data = np.load(prob_file,allow_pickle=True).item()[values[0]]

                            if ratio != 0:
                                zeros_pad_gussion = np.zeros(int(frame_num * rate * ratio), dtype=float)
                                npyloader.append(zeros_pad * int(frame_num * rate * ratio) + load_data)
                                gussion_weights.append(np.concatenate(
                                    (zeros_pad_gussion, generate_gaussian_weights_with_stride(load_data,
                                                                                              length=cfg.DATA.NUM_FRAMES * rate,
                                                                                              stride=cfg.DATA.NUM_FRAMES * rate))))
                            else:
                                npyloader.append(load_data)

                                gussion_weights.append(
                                    generate_gaussian_weights_with_stride(load_data,
                                                                          length=cfg.DATA.NUM_FRAMES * rate,
                                                                          stride=cfg.DATA.NUM_FRAMES * rate))
                            npynum += 1

            max_len = len(max(npyloader, key=len))
            for sq in npyloader:
                sq.extend(zeros_pad * (max_len - len(sq)))
            gussion_weights_new = []

            for sq in gussion_weights:
                gussion_weights_new.append(np.concatenate((sq, np.zeros(max_len - len(sq)))))

            if use_gussion:
                ## ensemble by gussion
                count_final = []
                for pred, gussion_weight in zip(npyloader, gussion_weights_new):
                    pred = np.array(pred)[:, 0]
                    gussion_weight = gussion_weight[:, None]
                    count_final.append(pred * gussion_weight)
                count_final = np.array(count_final)
                gussion_weights_new = np.array(gussion_weights_new)[:, :, None]
                final_result = (np.sum(count_final, axis=0) / np.sum(gussion_weights_new, axis=0))[:, None, :]
            else:
                ## ensemble by mean
                mask = np.ma.masked_where(npyloader == zeros_pad, npyloader)
                final_result = np.ma.mean(mask, axis=0)

            if view == 'dashboard':
                dashboard_result = final_result
            elif view == 'rearview':
                rearview_result = final_result
            elif view == 'right':
                right_result = final_result
            npyloader = []
            gussion_weights = []


        dashboard_result = dashboard_result.tolist()
        rearview_result = rearview_result.tolist()
        right_result = right_result.tolist()
        max_len = len(max([dashboard_result, rearview_result, right_result], key=len))
        for sq in [dashboard_result, rearview_result, right_result]:
            sq.extend(zeros_pad * (max_len - len(sq)))
        dashboard_result = np.array(dashboard_result)
        rearview_result = np.array(rearview_result)
        right_result = np.array(right_result)

        dashboard_weight = np.ones((1, 16))
        rearview_weight = np.ones((1, 16))
        right_weight = np.ones((1, 16))

        dashboard_class = [3, 4, 13, 15]
        rearview_class = [3, 4, 13, 15]
        rightview_class = [5, 8, 11, 15]
        for i in dashboard_class:
            dashboard_weight[0][i] = 1000
        for i in rearview_class:
            rearview_weight[0][i] = 1000
        for i in rightview_class:
            right_weight[0][i] = 1000
            if i == 15:
                right_weight[0][i] = 500

        # weight normalization
        sum_weight = dashboard_weight + rearview_weight + right_weight
        dashboard_weight /= sum_weight
        rearview_weight /= sum_weight
        right_weight /= sum_weight

        # weighted summation
        dashboard_result *= dashboard_weight
        rearview_result *= rearview_weight
        right_result *= right_weight

        fusion_result = dashboard_result + rearview_result + right_result

        total_prob_sq[values[0]] = fusion_result

    return dict(sorted(total_prob_sq.items())), video_order


def get_classification(sequence_class_prob):
    classify = [[x, y] for x, y in zip(np.argmax(sequence_class_prob, axis=1), np.max(sequence_class_prob, axis=1))]
    labels_index = np.argmax(sequence_class_prob, axis=1)  # returns list of position of max value in each list.
    probs = np.max(sequence_class_prob, axis=1)  # return list of max value in each  list.
    return labels_index, probs


def activity_localization(prob_sq):
    action_idx, action_probs = get_classification(prob_sq)
    ts = 0.5
    class_confidence = {i: [0, 0] for i in range(0, 16)}

    for i in range(len(action_idx)):
        class_confidence[action_idx[i]][0] += action_probs[i]
        class_confidence[action_idx[i]][1] += 1

    result = []
    for key in sorted(class_confidence.keys()):
        if class_confidence[key][1] > 0:
            result.append(class_confidence[key][0] / class_confidence[key][1])
        else:
            result.append(ts)
    result = [i - 0.1 for i in result]
    # result = [i * 0.5 for i in result]
    result = [i if i > 0.1 else 0.1 for i in result]
    threshold = [ts if x > ts else x for x in result]

    action_tag = np.array([1 if action_probs[i] > threshold[action_idx[i]] else 0 for i in range(len(action_probs))])

    # print('action_tag', action_tag)
    activities_idx = []
    startings = []
    endings = []

    action_probility = []

    for i in range(len(action_tag)):
        if action_tag[i] == 1:
            activities_idx.append(action_idx[i])
            action_probility.append(action_probs[i])

            start = i
            end = i + 1
            startings.append(start)
            endings.append(end)

    return activities_idx, startings, endings, action_probility


def merge_and_remove(data):
    df_total = pd.DataFrame([[0, 0, 0, 0, 0]], columns=[0, 1, 2, 3, 4])
    print('df_total', df_total)
    for i in range(1, 11):
        # print(i)
        data_video = data[data[0] == i]
        print(data_video)
        list_label = data_video[1].unique()
        print(list_label)
        for label in list_label:
            data_video_label = data_video[data_video[1] == label]
            data_video_label = data_video_label.reset_index()
            print('data_video_label')
            countrow = 1
            for j in range(len(data_video_label) - 1):
                if data_video_label.loc[j + 1, 2] - data_video_label.loc[j, 3] <= 8:
                    data_video_label.loc[j + 1, 2] = data_video_label.loc[j, 2]
                    data_video_label.loc[j, 3] = 0
                    data_video_label.loc[j, 2] = 0
            for j in range(len(data_video_label)):
                if data_video_label.loc[j, 2] == 0 and data_video_label.loc[j, 3] == 0:
                    countrow += 1
                    data_video_label.loc[j + 1, 4] += data_video_label.loc[j, 4]
                    data_video_label.loc[j, 4] = 0
                else:
                    data_video_label.loc[j, 4] = data_video_label.loc[j, 4] / countrow
                    countrow = 1

            df_total = df_total.append(data_video_label)

    df_total = df_total[df_total[3] != 0]
    df_total = df_total[1 < df_total[3] - df_total[2]]
    df_total = df_total[df_total[3] - df_total[2] < 30]

    df_total = df_total.sort_values(by=[0, 2])

    drop_index = []
    for num in range(len(df_total) - 1):
        row = df_total.iloc[num]
        next_row = df_total.iloc[num + 1]
        if next_row[2] - row[3] <= 5:
            if (row[1] == 2 or row[1] == 3) and (next_row[1] == 5 or next_row[1] == 6):
                # row[3] = next_row[3]
                drop_index.append(df_total.iloc[num + 1]['index'])
                # df_total.drop(index=df_total.index[num+1],inplace=True)
            elif (row[1] == 5 or row[1] == 6) and (next_row[1] == 2 or next_row[1] == 3):
                # next_row[2] = row[2]
                drop_index.append(df_total.iloc[num]['index'])
                # df_total.drop(index=df_total.index[num],inplace=True)

            if (row[1] == 14 and next_row[1] == 3):
                df_total.iloc[num, 2] = min(df_total.iloc[num, 2], df_total.iloc[num + 1, 2])
                df_total.iloc[num, 3] = max(df_total.iloc[num, 3], df_total.iloc[num + 1, 3])
                drop_index.append(df_total.iloc[num + 1]['index'])
            elif (next_row[1] == 14 and row[1] == 3):
                df_total.iloc[num + 1, 2] = min(df_total.iloc[num, 2], df_total.iloc[num + 1, 2])
                df_total.iloc[num + 1, 3] = max(df_total.iloc[num, 3], df_total.iloc[num + 1, 3])
                drop_index.append(df_total.iloc[num]['index'])
            # print('test')
            #
            if (row[1] == 11 and next_row[1] == 12):
                df_total.iloc[num, 2] = min(df_total.iloc[num, 2], df_total.iloc[num + 1, 2])
                df_total.iloc[num, 3] = max(df_total.iloc[num, 3], df_total.iloc[num + 1, 3])
                drop_index.append(df_total.iloc[num + 1]['index'])
            elif (next_row[1] == 11 and row[1] == 12):
                df_total.iloc[num + 1, 2] = min(df_total.iloc[num, 2], df_total.iloc[num + 1, 2])
                df_total.iloc[num + 1, 3] = max(df_total.iloc[num, 3], df_total.iloc[num + 1, 3])
                drop_index.append(df_total.iloc[num]['index'])

    drop_index = list(set(drop_index))
    for d_index in drop_index:
        # df_total.drop(index=d_index,inplace=True)
        df_total = df_total[df_total['index'] != d_index]
    final_result = pd.DataFrame([[0, 0, 0, 0, 0]], columns=[0, 1, 2, 3, 4])
    for i in range(1, 11):
        data_video = df_total[df_total[0] == i]
        print(data_video)
        list_label = data_video[1].unique()
        for label in list_label:
            data_video_label = data_video[data_video[1] == label]
            data_video_label = data_video_label.reset_index()

            if len(data_video_label) == 0:
                continue
            elif len(data_video_label) == 1:
                append_item = data_video_label.iloc[0]
            else:
                # maxprob_index = data_video_label[4].idxmax()
                maxprob_index = data_video_label[4].idxmax()

                second_maxprob_index = data_video_label[4].drop(maxprob_index).idxmax()
                if data_video_label.loc[second_maxprob_index][4] > 0.95 * data_video_label.loc[maxprob_index][4]:
                    append_item_second = data_video_label.loc[second_maxprob_index]
                    final_result = final_result.append(append_item_second)

                append_item = data_video_label.loc[maxprob_index]

            data_video_label = append_item
            final_result = final_result.append(data_video_label)

    final_result = final_result.drop(4, axis=1)
    final_result = final_result.iloc[1:]
    final_result = final_result.drop(columns=['level_0', 'index'])
    final_result = final_result.astype('int').sort_values(by=[0, 2])
    df_total = df_total.drop(columns=['index'])

    df_total = final_result

    save_path = './output/final_submission.txt'
    df_total.to_csv(save_path, sep=' ', index=False, header=False)

def general_submission(data):
    # data = pd.read_csv(filename, sep=" ", header=None)
    print(data)
    # data : [i,lable,start,end]
    data_filtered = data[data[1] != 0]
    # data_filtered = data
    print(data_filtered)
    data_filtered[2] = data[2].map(lambda x: int(float(x)))
    data_filtered[3] = data[3].map(lambda x: int(float(x)))

    data_filtered = data_filtered.sort_values(by=[0, 1])
    print(data_filtered)
    merge_and_remove(data_filtered)
    # return True


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)
    fps = 30
    seed_everything(719)
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    video_ids = {}  # saving the videos
    video_names = []
    path = cfg.DATA.PATH_TO_DATA_DIR
    print(path)
    with open(os.path.join(path, 'video_ids.csv')) as csvfile:
        csvReader = csv.reader(csvfile, delimiter=',')
        for idx, row in enumerate(csvReader):
            if idx > 0:
                video_ids[row[1]] = row[0]
                video_names.append(row[1])
    # path = './data/A2/'
    import glob

    text_files = glob.glob(path + "/**/*.MP4", recursive=True)
    filelist = {}
    for root, dirs, files in os.walk(path):
        for vid_name in files:  # Loop over directories, not files
            if vid_name in video_names:  # Only keep ones that match
                filelist[vid_name] = os.path.join(root, vid_name)
    vid_info = {}
    for key in (video_ids.keys() | filelist.keys()):
        if key in video_ids: vid_info.setdefault(key, []).append(video_ids[key])
        if key in filelist: vid_info.setdefault(key, []).append(filelist[key])

    vid_info = dict(sorted(vid_info.items()))
    prob_ensemble, video_order = ensemble_results(cfg, vid_info)

    dataframe_list = []

    # iterate each video
    for i in range(1, len(vid_info) + 1):
        len_prob = len(prob_ensemble[str(i)])
        prob_ensemble_video = []
        for ids in range(len_prob):
            prob_sub_mean = prob_ensemble[str(i)][ids]
            prob_ensemble_video.append(prob_sub_mean)

        #### post processing
        print('post-processing output....')
        prob_actions = np.array(prob_ensemble_video)
        prob_actions = np.squeeze(prob_actions)

        # ======Temporal localization==========
        activities_idx, startings, endings, activities_probility = activity_localization(prob_actions)
        print('\Results:')
        print('Video_id\tLabel\tInterval\t\tActivity')
        for idx, s, e, p in zip(activities_idx, startings, endings, activities_probility):
            start = s / fps
            end = e / fps
            label = labels[idx]
            print(
                '{}\t{}\t{:.1f}s - {:.1f}s\t'.format(i, label, start, end, p))
            dataframe_list.append([i, label, start, end, p])

    data = pd.DataFrame(dataframe_list, columns=[0, 1, 2, 3, 4])
    general_submission(data)
