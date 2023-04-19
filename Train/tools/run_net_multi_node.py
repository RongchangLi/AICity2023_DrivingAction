#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Wrapper to train and test a video classification model."""
from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args

# from demo_net import demo
from test_net import test
from train_net import train
# from visualization import visualize

import os
from pathlib import Path

def parse_ip(s):
    s = s.split("-")
    s = [y for x in s for y in x.split("[") if y]
    s = [y for x in s for y in x.split(",") if y ]

    return ".".join(s[2:6])


def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()

    if 'SLURM_STEP_NODELIST' in os.environ:
        args.init_method = "tcp://{}:{}".format(
            parse_ip(os.environ['SLURM_STEP_NODELIST']), "9999")

        print("Init Method: {}".format(args.init_method))

        cfg = load_config(args)
        cfg = assert_and_infer_cfg(cfg)

        cfg.NUM_SHARDS = int(os.environ['SLURM_NTASKS'])
        cfg.SHARD_ID = int(os.environ['SLURM_NODEID'])

        print(f'node id > {cfg.SHARD_ID}')
    else:
        cfg = load_config(args)
        cfg = assert_and_infer_cfg(cfg)
    out_dir=cfg.OUTPUT_DIR
    # Perform training.
    if cfg.TRAIN.ENABLE or cfg.TEST.ENABLE:
        for seed in [
            1,
            # 128,256,
            # 512, 1024
        ]:
            for manner in [
                "original_zero",
                            "expand_zero"
                           ]:
                for split in [
                    'new_split_by_id_A1_total',
                    #           'new_split_by_id_A1_20'
                              ]:
                    for view in [
                        "dashboard",
                        "rearview",
                        "right",
                        # "all"
                    ]:
                        if 'expand' in manner:
                            cfg.TRAIN.CHECKPOINT_FILE_PATH.replace('original','expand')
                        cfg.view = view
                        cfg.manner = manner
                        cfg.RNG_SEED = seed
                        cfg.split = split
                        frames = cfg.DATA.NUM_FRAMES
                        s_rate = cfg.DATA.SAMPLING_RATE
                        arch = cfg.MODEL.ARCH
                        input_size = cfg.DATA.TRAIN_CROP_SIZE
                        cfg.OUTPUT_DIR = os.path.join(out_dir,'{}_{}_{}_f{}_r{}_is{}/checkpoint_{}_{}'.format(split, cfg.RNG_SEED, arch,
                                                                                             frames, s_rate, input_size,
                                                                                             view, manner))

                        # Path(os.path.join(working_dir, 'visual')).mkdir(parents=True, exist_ok=True)
                        if not os.path.exists(cfg.OUTPUT_DIR):
                            Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
                        if cfg.TRAIN.ENABLE:
                            launch_job(cfg=cfg, init_method=args.init_method, func=train)

                        # Perform multi-clip testing.
                        if cfg.TEST.ENABLE:
                            launch_job(cfg=cfg, init_method=args.init_method, func=test)

                        # # Perform model visualization.
                        # if cfg.TENSORBOARD.ENABLE and (
                        #     cfg.TENSORBOARD.MODEL_VIS.ENABLE
                        #     or cfg.TENSORBOARD.WRONG_PRED_VIS.ENABLE
                        # ):
                        #     launch_job(cfg=cfg, init_method=args.init_method, func=visualize)

                        # Run demo.
                        # if cfg.DEMO.ENABLE:
                        #     demo(cfg)


if __name__ == "__main__":
    main()
