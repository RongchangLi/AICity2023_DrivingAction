# !/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
import os
import torch
import random
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
from slowfast.models import build_model
import cv2
import glob

"""Wrapper to train and test a video classification model."""
from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args
import time

logger = logging.get_logger(__name__)
import csv
from itertools import islice


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


@torch.no_grad()
def inference_result(cfg, videoids, labels, path, checkpoint_list):
    """
    Main function to spawn the train and test process.
    """
    print(videoids)
    # print("CFG: ", cfg)
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    # print('cfg', cfg.TEST.CHECKPOINT_FILE_PATH)
    for checki in checkpoint_list:
        cfg.TEST.CHECKPOINT_FILE_PATH = checki
        datatype = checki.split('/')[-1].split('_')[0]
        viewtype = checki.split('/')[-1].split('_')[1]
        cfg.DATA.NUM_FRAMES = int(checki.split('/')[-1].split('_')[2].split('frame')[1])
        cfg.DATA.SAMPLING_RATE = int(checki.split('/')[-1].split('_')[3].split('rate')[1].split('.')[0])
        window_start = cfg.DATA.NUM_FRAMES * cfg.DATA.SAMPLING_RATE / 4
        model = build_model(cfg)
        cu.load_test_checkpoint(cfg, model)
        model.eval()
        print('cfg', cfg.TEST.CHECKPOINT_FILE_PATH)

        total_prob_sq1 = {}
        total_prob_sq2 = {}
        total_prob_sq3 = {}
        total_prob_sq4 = {}
        video_order = []
        for key, values in videoids.items():
            video_order.append(values)
            video_path = values[1]
            print(video_path)
            img_provider = VideoReader(video_path)
            fps = 30
            print('fps:', fps)
            frames = []
            frames2 = []
            frames3 = []
            frames4 = []

            count = 0
            count2 = 0
            count3 = 0
            count4 = 0

            prob_sq1 = []
            prob_sq2 = []
            prob_sq3 = []
            prob_sq4 = []
            overlapratio = [0, 0.25, 0.5, 0.75]
            for able_to_read, frame in img_provider:
                count += 1
                # i += 1
                if not able_to_read:
                    break
                if len(frames) != cfg.DATA.NUM_FRAMES and count % cfg.DATA.SAMPLING_RATE == 0 and count > 0 *cfg.DATA.NUM_FRAMES*cfg.DATA.SAMPLING_RATE:
                    frame_processed = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # 先crop再resize
                    frame_processed = cv2.resize(frame_processed, (512, 512), interpolation=cv2.INTER_AREA)
                    frames.append(frame_processed)
                if len(frames) == cfg.DATA.NUM_FRAMES:
                    start = time.time()
                    # Perform color normalization.
                    inputs = torch.tensor(np.array(frames)).float()
                    inputs = inputs / 255.0
                    inputs = inputs - torch.tensor(cfg.DATA.MEAN)
                    inputs = inputs / torch.tensor(cfg.DATA.STD)
                    # print(cfg.DATA.MEAN, cfg.DATA.STD)
                    #
                    # T H W C -> C T H W.
                    inputs = inputs.permute(3, 0, 1, 2)
                    # 1 C T H W.  1*3*8*512*512
                    inputs = inputs[None, :, :, :, :]
                    # Sample frames for the fast pathway.
                    index = torch.linspace(0, inputs.shape[2] - 1, cfg.DATA.NUM_FRAMES).long()
                    fast_pathway = torch.index_select(inputs, 2, index)
                    inputs = [inputs]
                    # Transfer the data to the current GPU device.
                    if isinstance(inputs, (list,)):
                        for i in range(len(inputs)):
                            inputs[i] = inputs[i].cuda(non_blocking=True)
                    else:
                        inputs = inputs.cuda(non_blocking=True)
                    # print('inputs[0].shape', inputs[0].shape)
                    # Perform the forward pass.
                    preds = model(inputs).detach().cpu().numpy()
                    prob_ensemble = np.array([preds])
                    prob_ensemble = np.mean(prob_ensemble, axis=0)
                    for i in range(cfg.DATA.NUM_FRAMES*cfg.DATA.SAMPLING_RATE):
                        prob_sq1.append(prob_ensemble)
                    frames = []

            for able_to_read, frame in img_provider:
                count2 += 1
                if not able_to_read:
                    break
                if len(frames2) != cfg.DATA.NUM_FRAMES and count2 % cfg.DATA.SAMPLING_RATE == 0 and count2 > 0.25 *cfg.DATA.NUM_FRAMES*cfg.DATA.SAMPLING_RATE:
                    frame_processed2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_processed2 = cv2.resize(frame_processed2, (512, 512), interpolation=cv2.INTER_AREA)
                    frames2.append(frame_processed2)
                if len(frames2) == cfg.DATA.NUM_FRAMES:
                    start = time.time()
                    # Perform color normalization.
                    inputs = torch.tensor(np.array(frames2)).float()
                    inputs = inputs / 255.0
                    inputs = inputs - torch.tensor(cfg.DATA.MEAN)
                    inputs = inputs / torch.tensor(cfg.DATA.STD)
                    # print(cfg.DATA.MEAN, cfg.DATA.STD)
                    #
                    # T H W C -> C T H W.
                    inputs = inputs.permute(3, 0, 1, 2)
                    # 1 C T H W.  1*3*8*512*512
                    inputs = inputs[None, :, :, :, :]
                    # Sample frames for the fast pathway.
                    index = torch.linspace(0, inputs.shape[2] - 1, cfg.DATA.NUM_FRAMES).long()
                    fast_pathway = torch.index_select(inputs, 2, index)
                    inputs = [inputs]
                    # Transfer the data to the current GPU device.
                    if isinstance(inputs, (list,)):
                        for i in range(len(inputs)):
                            inputs[i] = inputs[i].cuda(non_blocking=True)
                    else:
                        inputs = inputs.cuda(non_blocking=True)
                    preds_2 = model(inputs).detach().cpu().numpy()
                    prob_ensemble = np.array([preds_2])
                    prob_ensemble = np.mean(prob_ensemble, axis=0)

                    for i in range(cfg.DATA.NUM_FRAMES*cfg.DATA.SAMPLING_RATE):
                        prob_sq2.append(prob_ensemble)
                    frames2 = []

            for able_to_read, frame in img_provider:
                count3 += 1
                if not able_to_read:
                    break
                if len(frames3) != cfg.DATA.NUM_FRAMES and count3 % cfg.DATA.SAMPLING_RATE == 0 and count3 > 0.5 *cfg.DATA.NUM_FRAMES*cfg.DATA.SAMPLING_RATE:
                    frame_processed3 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_processed3 = cv2.resize(frame_processed3, (512, 512), interpolation=cv2.INTER_AREA)
                    frames3.append(frame_processed3)

                if len(frames3) == cfg.DATA.NUM_FRAMES:

                    start = time.time()
                    # Perform color normalization.
                    inputs = torch.tensor(np.array(frames3)).float()
                    inputs = inputs / 255.0
                    inputs = inputs - torch.tensor(cfg.DATA.MEAN)
                    inputs = inputs / torch.tensor(cfg.DATA.STD)
                    # print(cfg.DATA.MEAN, cfg.DATA.STD)
                    #
                    # T H W C -> C T H W.
                    inputs = inputs.permute(3, 0, 1, 2)
                    # 1 C T H W.  1*3*8*512*512
                    inputs = inputs[None, :, :, :, :]
                    # Sample frames for the fast pathway.
                    index = torch.linspace(0, inputs.shape[2] - 1, cfg.DATA.NUM_FRAMES).long()
                    fast_pathway = torch.index_select(inputs, 2, index)
                    inputs = [inputs]
                    # Transfer the data to the current GPU device.
                    if isinstance(inputs, (list,)):
                        for i in range(len(inputs)):
                            inputs[i] = inputs[i].cuda(non_blocking=True)
                    else:
                        inputs = inputs.cuda(non_blocking=True)

                    preds_3 = model(inputs).detach().cpu().numpy()
                    prob_ensemble = np.array([preds_3])
                    prob_ensemble = np.mean(prob_ensemble, axis=0)

                    for i in range(cfg.DATA.NUM_FRAMES * cfg.DATA.SAMPLING_RATE):
                        prob_sq3.append(prob_ensemble)
                    frames3 = []

            for able_to_read, frame in img_provider:
                count4 += 1
                if not able_to_read:
                    break
                if len(frames4) != cfg.DATA.NUM_FRAMES and count4 % cfg.DATA.SAMPLING_RATE == 0 and count3 > 0.75 *cfg.DATA.NUM_FRAMES*cfg.DATA.SAMPLING_RATE:
                    frame_processed4 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_processed4 = cv2.resize(frame_processed4, (512, 512), interpolation=cv2.INTER_AREA)
                    frames4.append(frame_processed4)

                if len(frames4) == cfg.DATA.NUM_FRAMES:
                    start = time.time()
                    # Perform color normalization.
                    inputs = torch.tensor(np.array(frames4)).float()
                    inputs = inputs / 255.0
                    inputs = inputs - torch.tensor(cfg.DATA.MEAN)
                    inputs = inputs / torch.tensor(cfg.DATA.STD)
                    # print(cfg.DATA.MEAN, cfg.DATA.STD)
                    #
                    # T H W C -> C T H W.
                    inputs = inputs.permute(3, 0, 1, 2)
                    # 1 C T H W.  1*3*8*512*512
                    inputs = inputs[None, :, :, :, :]
                    # Sample frames for the fast pathway.
                    index = torch.linspace(0, inputs.shape[2] - 1, cfg.DATA.NUM_FRAMES).long()
                    fast_pathway = torch.index_select(inputs, 2, index)
                    inputs = [inputs]
                    # Transfer the data to the current GPU device.
                    if isinstance(inputs, (list,)):
                        for i in range(len(inputs)):
                            inputs[i] = inputs[i].cuda(non_blocking=True)
                    else:
                        inputs = inputs.cuda(non_blocking=True)

                    preds_4 = model(inputs).detach().cpu().numpy()
                    prob_ensemble = np.array([preds_4])
                    prob_ensemble = np.mean(prob_ensemble, axis=0)
                    for i in range(cfg.DATA.NUM_FRAMES * cfg.DATA.SAMPLING_RATE):
                        prob_sq4.append(prob_ensemble)
                    frames4 = []


            total_prob_sq1[values[0]] = prob_sq1
            total_prob_sq2[values[0]] = prob_sq2
            total_prob_sq3[values[0]] = prob_sq3
            total_prob_sq4[values[0]] = prob_sq4

        np.save('./probability_results/'
                'view-{}_frame-{}_rate-{}_datatype-{}_overlapratio-{}.npy'.format(viewtype,cfg.DATA.NUM_FRAMES,cfg.DATA.SAMPLING_RATE,datatype, 0),dict(sorted(total_prob_sq1.items())))
        np.save('./probability_results/'
                'view-{}_frame-{}_rate-{}_datatype-{}_overlapratio-{}.npy'.format(viewtype,cfg.DATA.NUM_FRAMES,cfg.DATA.SAMPLING_RATE,datatype, 0.25),dict(sorted(total_prob_sq2.items())))
        np.save('./probability_results/'
                'view-{}_frame-{}_rate-{}_datatype-{}_overlapratio-{}.npy'.format(viewtype,cfg.DATA.NUM_FRAMES,cfg.DATA.SAMPLING_RATE,datatype, 0.5), dict(sorted(total_prob_sq3.items())))
        np.save('./probability_results/'
                'view-{}_frame-{}_rate-{}_datatype-{}_overlapratio-{}.npy'.format(viewtype,cfg.DATA.NUM_FRAMES,cfg.DATA.SAMPLING_RATE,datatype, 0.75), dict(sorted(total_prob_sq4.items())))


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
    path = cfg.DATA.PATH_TO_DATA_DIR
    print(path)

    video_ids = {}  # saving the videos
    video_names = []
    with open(os.path.join(path, 'video_ids.csv')) as csvfile:
        csvReader = csv.reader(csvfile, delimiter=',')
        for idx, row in enumerate(csvReader):
            if idx > 0:
                video_ids[row[1]] = row[0]
                video_names.append(row[1])
    # path = './data/A2/'
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
    # original: original action  expand: expand background
    checkpoint_dashboard_list = [
            './checkpoint_submit/original_dashboard_frame8_rate4.pyth',
            './checkpoint_submit/original_dashboard_frame8_rate8.pyth',
            './checkpoint_submit/original_dashboard_frame8_rate12.pyth',
            './checkpoint_submit/original_dashboard_frame16_rate2.pyth',
            './checkpoint_submit/original_dashboard_frame16_rate4.pyth',
            './checkpoint_submit/original_dashboard_frame16_rate6.pyth',

            './checkpoint_submit/expand_dashboard_frame8_rate4.pyth',
            './checkpoint_submit/expand_dashboard_frame8_rate8.pyth',
            './checkpoint_submit/expand_dashboard_frame8_rate12.pyth',
            './checkpoint_submit/expand_dashboard_frame16_rate2.pyth',
            './checkpoint_submit/expand_dashboard_frame16_rate4.pyth',
            './checkpoint_submit/expand_dashboard_frame16_rate6.pyth'
        ]
    vid_info = dict(sorted(vid_info.items()))
    inference_result(cfg, vid_info, labels, filelist, checkpoint_dashboard_list)

    video_ids = {}
    video_names = []
    with open(os.path.join(path, 'video_ids.csv')) as csvfile:
        csvReader = csv.reader(csvfile, delimiter=',')
        for idx, row in enumerate(csvReader):
            if idx > 0:
                video_ids[row[2]] = row[0]
                video_names.append(row[2])
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
    # rearview
    checkpoint_rearview_list = [
        './checkpoint_submit/original_rearview_frame8_rate4.pyth',
        './checkpoint_submit/original_rearview_frame8_rate8.pyth',
        './checkpoint_submit/original_rearview_frame8_rate12.pyth',
        './checkpoint_submit/original_rearview_frame16_rate2.pyth',
        './checkpoint_submit/original_rearview_frame16_rate4.pyth',
        './checkpoint_submit/original_rearview_frame16_rate6.pyth',

        './checkpoint_submit/expand_rearview_frame8_rate4.pyth',
        './checkpoint_submit/expand_rearview_frame8_rate8.pyth',
        './checkpoint_submit/expand_rearview_frame8_rate12.pyth',
        './checkpoint_submit/expand_rearview_frame16_rate2.pyth',
        './checkpoint_submit/expand_rearview_frame16_rate4.pyth',
        './checkpoint_submit/expand_rearview_frame16_rate6.pyth'
    ]
    inference_result(cfg, vid_info, labels, filelist, checkpoint_rearview_list)
    #
    video_ids = {}
    video_names = []
    with open(os.path.join(path, 'video_ids.csv')) as csvfile:
        csvReader = csv.reader(csvfile, delimiter=',')
        for idx, row in enumerate(csvReader):
            if idx > 0:
                video_ids[row[3]] = row[0]
                video_names.append(row[3])
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
    checkpoint_right_list = [
        './checkpoint_submit/original_right_frame8_rate4.pyth',
        './checkpoint_submit/original_right_frame8_rate8.pyth',
        './checkpoint_submit/original_right_frame8_rate12.pyth',
        './checkpoint_submit/original_right_frame16_rate2.pyth',
        './checkpoint_submit/original_right_frame16_rate4.pyth',
        './checkpoint_submit/original_right_frame16_rate6.pyth',

        './checkpoint_submit/expand_right_frame8_rate4.pyth',
        './checkpoint_submit/expand_right_frame8_rate8.pyth',
        './checkpoint_submit/expand_right_frame8_rate12.pyth',
        './checkpoint_submit/expand_right_frame16_rate2.pyth',
        './checkpoint_submit/expand_right_frame16_rate4.pyth',
        './checkpoint_submit/expand_right_frame16_rate6.pyth'
    ]
    inference_result(cfg, vid_info, labels, filelist, checkpoint_right_list)