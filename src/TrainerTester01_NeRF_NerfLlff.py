# Shree KRISHNAya Namaha
# Modified from VER006/NerfLlffTrainerTester01.py.
# Author: Nagabhushan S N
# Last Modified: 23/09/2022

import datetime
import os
import time
import traceback
from pathlib import Path

import numpy
import pandas
import skimage.io
import skvideo.io

import Tester01 as Tester
import Trainer01 as Trainer

this_filepath = Path(__file__)
this_filename = this_filepath.stem


def read_image(path: Path):
    image = skimage.io.imread(path.as_posix())
    return image


def save_video(path: Path, video: numpy.ndarray):
    if path.exists():
        return
    try:
        skvideo.io.vwrite(path.as_posix(), video,
                          inputdict={'-r': str(15)},
                          outputdict={'-c:v': 'libx264', '-pix_fmt': 'yuv420p'}, verbosity=1)
    except OSError:
        pass
    return


def start_training(train_configs: dict):
    root_dirpath = Path('../')
    project_dirpath = root_dirpath / '../../../../'
    database_dirpath = project_dirpath / train_configs['database_dirpath']

    # Setup output dirpath
    output_dirpath = root_dirpath / f'Runs/Training/Train{train_configs["train_num"]:04}'
    output_dirpath.mkdir(parents=True, exist_ok=True)
    Trainer.save_configs(output_dirpath, train_configs)

    if 'scene_ids' not in train_configs['data_loader']:
        set_num = train_configs['data_loader']['train_set_num']
        video_datapath = database_dirpath / f'TrainTestSets/Set{set_num:02}/TrainVideosData.csv'
        video_data = pandas.read_csv(video_datapath)
        scene_names = video_data['scene_name'].to_numpy()
        scene_ids = numpy.unique(scene_names)
        train_configs['data_loader']['scene_ids'] = scene_ids
    Trainer.start_training(train_configs)
    return


def start_testing(test_configs: dict):
    root_dirpath = Path('../')
    project_dirpath = root_dirpath / '../../../../'
    database_dirpath = project_dirpath / 'Databases' / test_configs['database_dirpath']

    output_dirpath = root_dirpath / f"Runs/Testing/Test{test_configs['test_num']:04}"
    output_dirpath.mkdir(parents=True, exist_ok=True)
    Tester.save_configs(output_dirpath, test_configs)

    set_num = test_configs['test_set_num']
    train_video_datapath = database_dirpath / f'TrainTestSets/Set{set_num:02}/TrainVideosData.csv'
    test_video_datapath = database_dirpath / f'TrainTestSets/Set{set_num:02}/TestVideosData.csv'
    train_video_data = pandas.read_csv(train_video_datapath)
    test_video_data = pandas.read_csv(test_video_datapath)
    scene_names = test_video_data['scene_name'].to_numpy()
    scene_ids = numpy.unique(scene_names)
    scenes_data = {}
    for scene_id in scene_ids:
        scene_name = scene_id
        scenes_data[scene_id] = {
            'output_dirname': scene_id,
            'frames_data': {}
        }

        extrinsics_path = database_dirpath / f'all/DatabaseData/{scene_name}/CameraExtrinsics.csv'
        extrinsics = numpy.loadtxt(extrinsics_path.as_posix(), delimiter=',').reshape((-1, 4, 4))

        frame_nums = test_video_data.loc[test_video_data['scene_name'] == scene_name]['pred_frame_num'].to_list() + \
                     train_video_data.loc[train_video_data['scene_name'] == scene_name]['pred_frame_num'].to_list()
        frame_nums = numpy.unique(sorted(frame_nums))
        for frame_num in frame_nums:
            scenes_data[scene_id]['frames_data'][frame_num] = {
                'extrinsic': extrinsics[frame_num]
            }
    Tester.start_testing(test_configs, scenes_data, save_depth=True, save_depth_var=True)

    # Run QA
    qa_filepath = Path('../../../../QA/00_Common/src/AllMetrics03_NeRF_LLFF.py')
    cmd = f'python {qa_filepath.absolute().as_posix()} ' \
          f'--demo_function_name demo2 ' \
          f'--pred_videos_dirpath {output_dirpath.absolute().as_posix()} ' \
          f'--database_dirpath {database_dirpath.absolute().as_posix()} ' \
          f'--frames_datapath {test_video_datapath.absolute().as_posix()} ' \
          f'--pred_folder_name PredictedFrames ' \
          f'--resolution_suffix _down4 '
    os.system(cmd)
    return


def start_testing_videos(test_configs: dict):
    root_dirpath = Path('../')
    project_dirpath = root_dirpath / '../../../../'
    database_dirpath = project_dirpath / 'Databases' / test_configs['database_dirpath']

    output_dirpath = root_dirpath / f"Runs/Testing/Test{test_configs['test_num']:04}"
    output_dirpath.mkdir(parents=True, exist_ok=True)
    Tester.save_configs(output_dirpath, test_configs)

    set_num = test_configs['test_set_num']
    video_datapath = database_dirpath / f'TrainTestSets/Set{set_num:02}/TestVideosData.csv'
    video_data = pandas.read_csv(video_datapath)
    scene_names = video_data['scene_name'].to_numpy()
    scene_ids = numpy.unique(scene_names)

    videos_data = [2, ]
    for video_num in videos_data:
        video_frame_nums_path = database_dirpath / f'TrainTestSets/Set{set_num:02}/VideoPoses{video_num:02}/VideoFrameNums.csv'
        if video_frame_nums_path.exists():
            video_frame_nums = numpy.loadtxt(video_frame_nums_path.as_posix(), delimiter=',').astype(int)
        else:
            video_frame_nums = None
        for scene_id in scene_ids:
            scenes_data = {}
            scene_name = scene_id
            scenes_data[scene_id] = {
                'output_dirname': scene_id,
                'frames_data': {}
            }

            extrinsics_path = database_dirpath / f'TrainTestSets/Set{set_num:02}/VideoPoses{video_num:02}/{scene_name}.csv'
            extrinsics = numpy.loadtxt(extrinsics_path.as_posix(), delimiter=',').reshape((-1, 4, 4))

            frame_nums = numpy.arange(extrinsics.shape[0] - 1)
            for frame_num in frame_nums:
                scenes_data[scene_id]['frames_data'][frame_num] = {
                    'extrinsic': extrinsics[frame_num + 1]
                }
            output_dir_suffix = f'_Video{video_num:02}'
            output_dirpath = Tester.start_testing(test_configs, scenes_data, output_dir_suffix)
            scene_output_dirpath = output_dirpath / f'{scene_id}{output_dir_suffix}'
            if not scene_output_dirpath.exists():
                continue
            pred_frames = [read_image(scene_output_dirpath / f'PredictedFrames/{frame_num:04}.png') for frame_num in frame_nums]
            video_frames = numpy.stack(pred_frames)
            if video_frame_nums is not None:
                video_frames = video_frames[video_frame_nums]
            video_output_path = scene_output_dirpath / 'PredictedVideo.mp4'
            save_video(video_output_path, video_frames)
    return


def start_testing_static_videos(test_configs: dict):
    """
    This is for view_dirs visualization
    :param test_configs:
    :return:
    """
    root_dirpath = Path('../')
    project_dirpath = root_dirpath / '../../../../'
    database_dirpath = project_dirpath / 'Databases' / test_configs['database_dirpath']

    output_dirpath = root_dirpath / f"Runs/Testing/Test{test_configs['test_num']:04}"
    output_dirpath.mkdir(parents=True, exist_ok=True)
    Tester.save_configs(output_dirpath, test_configs)

    set_num = test_configs['test_set_num']
    video_datapath = database_dirpath / f'TrainTestSets/Set{set_num:02}/TestVideosData.csv'
    video_data = pandas.read_csv(video_datapath)
    scene_names = video_data['scene_name'].to_numpy()
    scene_ids = numpy.unique(scene_names)

    videos_data = [2, ]
    for video_num in videos_data:
        video_frame_nums_path = database_dirpath / f'TrainTestSets/Set{set_num:02}/VideoPoses{video_num:02}/VideoFrameNums.csv'
        if video_frame_nums_path.exists():
            video_frame_nums = numpy.loadtxt(video_frame_nums_path.as_posix(), delimiter=',').astype(int)
        else:
            video_frame_nums = None
        for scene_id in scene_ids:
            scenes_data = {}
            scene_name = scene_id
            scenes_data[scene_id] = {
                'output_dirname': scene_id,
                'frames_data': {}
            }

            extrinsics_path = database_dirpath / f'TrainTestSets/Set{set_num:02}/VideoPoses{video_num:02}/{scene_name}.csv'
            extrinsics = numpy.loadtxt(extrinsics_path.as_posix(), delimiter=',').reshape((-1, 4, 4))

            frame_nums = numpy.arange(extrinsics.shape[0] - 1)
            for frame_num in frame_nums:
                scenes_data[scene_id]['frames_data'][frame_num] = {
                    'extrinsic': extrinsics[0],
                    'extrinsic_viewcam': extrinsics[frame_num + 1]
                }
            output_dir_suffix = f'_Video{video_num:02}_StaticCamera'
            output_dirpath = Tester.start_testing(test_configs, scenes_data, output_dir_suffix)
            scene_output_dirpath = output_dirpath / f'{scene_id}{output_dir_suffix}'
            if not scene_output_dirpath.exists():
                continue
            pred_frames = [read_image(scene_output_dirpath / f'PredictedFrames/{frame_num:04}.png') for frame_num in frame_nums]
            video_frames = numpy.stack(pred_frames)
            if video_frame_nums is not None:
                video_frames = video_frames[video_frame_nums]
            video_output_path = scene_output_dirpath / 'StaticCameraVideo.mp4'
            save_video(video_output_path, video_frames)
    return


def demo1():
    train_num = 11
    test_num = 11

    train_configs = {
        'trainer': f'{this_filename}/{Trainer.this_filename}',
        'train_num': train_num,
        'database': 'NeRF_LLFF',
        'database_dirpath': 'Databases/NeRF_LLFF/Data',
        'data_loader': {
            'data_loader_name': 'NerfLlffDataLoader01',
            'data_preprocessor_name': 'NeRFDataPreprocessor01',
            'train_set_num': 3,
            'scene_ids': ['fern', 'flower', 'fortress', 'horns', 'leaves', 'orchids', 'room', 'trex'],
            # 'scene_ids': ['room', ],
            'resolution_suffix': '_down4',
            'recenter_camera_poses': True,
            'bd_factor': 0.75,
            'spherify': False,
            'ndc': True,
            'batching': True,
            'num_rays': 1024,
        },
        'model': {
            'name': 'NeRF01',
            'use_coarse_mlp': True,
            'use_fine_mlp': True,
            'num_samples_coarse': 64,
            'num_samples_fine': 128,
            'chunk': 4*1024,
            'lindisp': False,
            'points_positional_encoding_degree': 10,
            'views_positional_encoding_degree': 4,
            'netchunk': 16*1024,
            'netdepth_coarse': 8,
            'netdepth_fine': 8,
            'netwidth_coarse': 256,
            'netwidth_fine': 256,
            'perturb': True,
            'raw_noise_std': 1.0,
            'use_view_dirs': True,
            'view_dependent_rgb': True,
            'white_bkgd': False,
        },
        'losses': [
            {
                'name': 'NeRF_MSE01',
                'weight': 1,
            },
        ],
        'optimizer': {
            'lr_decayer_name': 'NeRFLearningRateDecayer01',
            'lr_initial': 0.0005,
            'lr_decay': 250,
            'beta1': 0.9,
            'beta2': 0.999,
        },
        'resume_training': True,
        'num_iterations': 300000,
        'validation_interval': 500000,
        'num_validation_iterations': 10,
        'sample_save_interval': 500000,
        'model_save_interval': 25000,
        'mixed_precision_training': False,
        'seed': numpy.random.randint(1000),
        'device': 'gpu0',
    }
    test_configs = {
        'Tester': f'{this_filename}/{Tester.this_filename}',
        'test_num': test_num,
        'test_set_num': 3,
        'train_num': train_num,
        'model_name': 'Model_Iter300000.tar',
        'database_name': 'NeRF_LLFF',
        'database_dirpath': 'NeRF_LLFF/Data',
        'device': 'gpu0',
    }
    start_training(train_configs)
    start_testing(test_configs)
    start_testing_videos(test_configs)
    start_testing_static_videos(test_configs)
    return


def demo2():
    train_num = 12
    test_num = 12

    train_configs = {
        'trainer': f'{this_filename}/{Trainer.this_filename}',
        'train_num': train_num,
        'database': 'NeRF_LLFF',
        'database_dirpath': 'Databases/NeRF_LLFF/Data',
        'data_loader': {
            'data_loader_name': 'NerfLlffDataLoader01',
            'data_preprocessor_name': 'NeRFDataPreprocessor01',
            'train_set_num': 1,
            'scene_ids': ['fern', 'flower', 'fortress', 'horns', 'leaves', 'orchids', 'room', 'trex'],
            # 'scene_ids': ['room', ],
            'resolution_suffix': '_down4',
            'recenter_camera_poses': True,
            'bd_factor': 0.75,
            'spherify': False,
            'ndc': True,
            'batching': True,
            'num_rays': 1024,
            'precrop_fraction': 1,
            'precrop_iterations': -1,
        },
        'model': {
            'name': 'NeRF01',
            'use_coarse_mlp': True,
            'use_fine_mlp': True,
            'num_samples_coarse': 64,
            'num_samples_fine': 128,
            'chunk': 4*1024,
            'lindisp': False,
            'points_positional_encoding_degree': 10,
            'views_positional_encoding_degree': 4,
            'netchunk': 16*1024,
            'netdepth_coarse': 8,
            'netdepth_fine': 8,
            'netwidth_coarse': 256,
            'netwidth_fine': 256,
            'perturb': True,
            'raw_noise_std': 1.0,
            'use_view_dirs': True,
            'view_dependent_rgb': True,
            'white_bkgd': False,
        },
        'losses': [
            {
                'name': 'NeRF_MSE01',
                'weight': 1,
            },
        ],
        'optimizer': {
            'lr_decayer_name': 'NeRFLearningRateDecayer01',
            'lr_initial': 0.0005,
            'lr_decay': 250,
            'beta1': 0.9,
            'beta2': 0.999,
        },
        'resume_training': True,
        'num_iterations': 50000,
        'validation_interval': 500000,
        'num_validation_iterations': 10,
        'sample_save_interval': 500000,
        'model_save_interval': 25000,
        'mixed_precision_training': False,
        'seed': numpy.random.randint(1000),
        'device': 'gpu0',
    }
    test_configs = {
        'Tester': f'{this_filename}/{Tester.this_filename}',
        'test_num': test_num,
        'test_set_num': 1,
        'train_num': train_num,
        'model_name': 'Model_Iter050000.tar',
        'database_name': 'NeRF_LLFF',
        'database_dirpath': 'NeRF_LLFF/Data',
        'device': 'gpu0',
    }
    start_training(train_configs)
    start_testing(test_configs)
    start_testing_videos(test_configs)
    start_testing_static_videos(test_configs)
    return


def demo3():
    train_num = 13
    test_num = 13

    train_configs = {
        'trainer': f'{this_filename}/{Trainer.this_filename}',
        'train_num': train_num,
        'database': 'NeRF_LLFF',
        'database_dirpath': 'Databases/NeRF_LLFF/Data',
        'data_loader': {
            'data_loader_name': 'NerfLlffDataLoader01',
            'data_preprocessor_name': 'NeRFDataPreprocessor01',
            'train_set_num': 4,
            'scene_ids': ['fern', 'flower', 'fortress', 'horns', 'leaves', 'orchids', 'room', 'trex'],
            # 'scene_ids': ['room', ],
            'resolution_suffix': '_down4',
            'recenter_camera_poses': True,
            'bd_factor': 0.75,
            'spherify': False,
            'ndc': True,
            'batching': True,
            'num_rays': 1024,
            'precrop_fraction': 1,
            'precrop_iterations': -1,
        },
        'model': {
            'name': 'NeRF01',
            'use_coarse_mlp': True,
            'use_fine_mlp': True,
            'num_samples_coarse': 64,
            'num_samples_fine': 128,
            'chunk': 4*1024,
            'lindisp': False,
            'points_positional_encoding_degree': 10,
            'views_positional_encoding_degree': 4,
            'netchunk': 16*1024,
            'netdepth_coarse': 8,
            'netdepth_fine': 8,
            'netwidth_coarse': 256,
            'netwidth_fine': 256,
            'perturb': True,
            'raw_noise_std': 1.0,
            'use_view_dirs': True,
            'view_dependent_rgb': True,
            'white_bkgd': False,
        },
        'losses': [
            {
                'name': 'NeRF_MSE01',
                'weight': 1,
            },
        ],
        'optimizer': {
            'lr_decayer_name': 'NeRFLearningRateDecayer01',
            'lr_initial': 0.0005,
            'lr_decay': 250,
            'beta1': 0.9,
            'beta2': 0.999,
        },
        'resume_training': True,
        'num_iterations': 50000,
        'validation_interval': 500000,
        'num_validation_iterations': 10,
        'sample_save_interval': 500000,
        'model_save_interval': 25000,
        'mixed_precision_training': False,
        'seed': numpy.random.randint(1000),
        'device': 'gpu0',
    }
    test_configs = {
        'Tester': f'{this_filename}/{Tester.this_filename}',
        'test_num': test_num,
        'test_set_num': 4,
        'train_num': train_num,
        'model_name': 'Model_Iter050000.tar',
        'database_name': 'NeRF_LLFF',
        'database_dirpath': 'NeRF_LLFF/Data',
        'device': 'gpu0',
    }
    start_training(train_configs)
    start_testing(test_configs)
    start_testing_videos(test_configs)
    start_testing_static_videos(test_configs)
    return


def demo4():
    train_num = 14
    test_num = 14

    train_configs = {
        'trainer': f'{this_filename}/{Trainer.this_filename}',
        'train_num': train_num,
        'database': 'NeRF_LLFF',
        'database_dirpath': 'Databases/NeRF_LLFF/Data',
        'data_loader': {
            'data_loader_name': 'NerfLlffDataLoader01',
            'data_preprocessor_name': 'NeRFDataPreprocessor01',
            'train_set_num': 5,
            'scene_ids': ['fern', 'flower', 'fortress', 'horns', 'leaves', 'orchids', 'room', 'trex'],
            # 'scene_ids': ['room', ],
            'resolution_suffix': '_down4',
            'recenter_camera_poses': True,
            'bd_factor': 0.75,
            'spherify': False,
            'ndc': True,
            'batching': True,
            'num_rays': 1024,
            'precrop_fraction': 1,
            'precrop_iterations': -1,
        },
        'model': {
            'name': 'NeRF01',
            'use_coarse_mlp': True,
            'use_fine_mlp': True,
            'num_samples_coarse': 64,
            'num_samples_fine': 128,
            'chunk': 4*1024,
            'lindisp': False,
            'points_positional_encoding_degree': 10,
            'views_positional_encoding_degree': 4,
            'netchunk': 16*1024,
            'netdepth_coarse': 8,
            'netdepth_fine': 8,
            'netwidth_coarse': 256,
            'netwidth_fine': 256,
            'perturb': True,
            'raw_noise_std': 1.0,
            'use_view_dirs': True,
            'view_dependent_rgb': True,
            'white_bkgd': False,
        },
        'losses': [
            {
                'name': 'NeRF_MSE01',
                'weight': 1,
            },
        ],
        'optimizer': {
            'lr_decayer_name': 'NeRFLearningRateDecayer01',
            'lr_initial': 0.0005,
            'lr_decay': 250,
            'beta1': 0.9,
            'beta2': 0.999,
        },
        'resume_training': True,
        'num_iterations': 50000,
        'validation_interval': 500000,
        'num_validation_iterations': 10,
        'sample_save_interval': 500000,
        'model_save_interval': 25000,
        'mixed_precision_training': False,
        'seed': numpy.random.randint(1000),
        'device': 'gpu0',
    }
    test_configs = {
        'Tester': f'{this_filename}/{Tester.this_filename}',
        'test_num': test_num,
        'test_set_num': 5,
        'train_num': train_num,
        'model_name': 'Model_Iter050000.tar',
        'database_name': 'NeRF_LLFF',
        'database_dirpath': 'NeRF_LLFF/Data',
        'device': 'gpu0',
    }
    start_training(train_configs)
    start_testing(test_configs)
    start_testing_videos(test_configs)
    start_testing_static_videos(test_configs)
    return


def demo5():
    train_num = 15
    test_num = 15

    train_configs = {
        'trainer': f'{this_filename}/{Trainer.this_filename}',
        'train_num': train_num,
        'database': 'NeRF_LLFF',
        'database_dirpath': 'Databases/NeRF_LLFF/Data',
        'data_loader': {
            'data_loader_name': 'NerfLlffDataLoader01',
            'data_preprocessor_name': 'NeRFDataPreprocessor01',
            'train_set_num': 6,
            'scene_ids': ['fern', 'flower', 'fortress', 'horns', 'leaves', 'orchids', 'room', 'trex'],
            # 'scene_ids': ['room', ],
            'resolution_suffix': '_down4',
            'recenter_camera_poses': True,
            'bd_factor': 0.75,
            'spherify': False,
            'ndc': True,
            'batching': True,
            'downsampling_factor': 1,
            'num_rays': 1024,
            'precrop_fraction': 1,
            'precrop_iterations': -1,
        },
        'model': {
            'name': 'NeRF01',
            'use_coarse_mlp': True,
            'use_fine_mlp': True,
            'num_samples_coarse': 64,
            'num_samples_fine': 128,
            'chunk': 4*1024,
            'lindisp': False,
            'points_positional_encoding_degree': 10,
            'views_positional_encoding_degree': 4,
            'netchunk': 16*1024,
            'netdepth_coarse': 8,
            'netdepth_fine': 8,
            'netwidth_coarse': 256,
            'netwidth_fine': 256,
            'perturb': True,
            'raw_noise_std': 1.0,
            'use_view_dirs': True,
            'view_dependent_rgb': True,
            'white_bkgd': False,
        },
        'losses': [
            {
                'name': 'NeRF_MSE01',
                'weight': 1,
            },
        ],
        'optimizer': {
            'lr_decayer_name': 'NeRFLearningRateDecayer01',
            'lr_initial': 0.0005,
            'lr_decay': 250,
            'beta1': 0.9,
            'beta2': 0.999,
        },
        'resume_training': True,
        'num_iterations': 50000,
        'validation_interval': 10000,
        'validation_chunk_size': 64 * 1024,
        'validation_save_loss_maps': False,
        # 'num_validation_iterations': 10,
        # 'sample_save_interval': 10000,
        'model_save_interval': 25000,
        'mixed_precision_training': False,
        'seed': numpy.random.randint(1000),
        'device': 'gpu0',
    }
    test_configs = {
        'Tester': f'{this_filename}/{Tester.this_filename}',
        'test_num': test_num,
        'test_set_num': 6,
        'train_num': train_num,
        'model_name': 'Model_Iter050000.tar',
        'database_name': 'NeRF_LLFF',
        'database_dirpath': 'NeRF_LLFF/Data',
        'device': 'gpu0',
    }
    start_training(train_configs)
    start_testing(test_configs)
    start_testing_videos(test_configs)
    start_testing_static_videos(test_configs)
    return


def demo6():
    train_num = 16
    test_num = 16

    train_configs = {
        'trainer': f'{this_filename}/{Trainer.this_filename}',
        'train_num': train_num,
        'database': 'NeRF_LLFF',
        'database_dirpath': 'Databases/NeRF_LLFF/Data',
        'data_loader': {
            'data_loader_name': 'NerfLlffDataLoader01',
            'data_preprocessor_name': 'NeRFDataPreprocessor01',
            'train_set_num': 3,
            'scene_ids': ['fern', 'flower', 'fortress', 'horns', 'leaves', 'orchids', 'room', 'trex'],
            # 'scene_ids': ['room', ],
            'resolution_suffix': '_down4',
            'recenter_camera_poses': True,
            'bd_factor': 0.75,
            'spherify': False,
            'ndc': False,
            'batching': True,
            'downsampling_factor': 1,
            'num_rays': 1024,
            'precrop_fraction': 1,
            'precrop_iterations': -1,
        },
        'model': {
            'name': 'NeRF01',
            'use_coarse_mlp': True,
            'use_fine_mlp': True,
            'num_samples_coarse': 64,
            'num_samples_fine': 128,
            'chunk': 4*1024,
            'lindisp': False,
            'points_positional_encoding_degree': 10,
            'views_positional_encoding_degree': 4,
            'netchunk': 16*1024,
            'netdepth_coarse': 8,
            'netdepth_fine': 8,
            'netwidth_coarse': 256,
            'netwidth_fine': 256,
            'perturb': True,
            'raw_noise_std': 1.0,
            'use_view_dirs': True,
            'view_dependent_rgb': True,
            'white_bkgd': False,
        },
        'losses': [
            {
                'name': 'NeRF_MSE01',
                'weight': 1,
            },
        ],
        'optimizer': {
            'lr_decayer_name': 'NeRFLearningRateDecayer01',
            'lr_initial': 0.0005,
            'lr_decay': 250,
            'beta1': 0.9,
            'beta2': 0.999,
        },
        'resume_training': True,
        'num_iterations': 300000,
        'validation_interval': 500000,
        'num_validation_iterations': 10,
        'sample_save_interval': 500000,
        'model_save_interval': 25000,
        'mixed_precision_training': False,
        'seed': numpy.random.randint(1000),
        'device': 'gpu0',
    }
    test_configs = {
        'Tester': f'{this_filename}/{Tester.this_filename}',
        'test_num': test_num,
        'test_set_num': 3,
        'train_num': train_num,
        'model_name': 'Model_Iter300000.tar',
        'database_name': 'NeRF_LLFF',
        'database_dirpath': 'NeRF_LLFF/Data',
        'device': 'gpu0',
    }
    start_training(train_configs)
    start_testing(test_configs)
    start_testing_videos(test_configs)
    start_testing_static_videos(test_configs)
    return


def main():
    demo1()
    demo2()
    demo3()
    demo4()
    demo5()
    demo6()
    return


if __name__ == '__main__':
    print('Program started at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    start_time = time.time()
    try:
        main()
        run_result = 'Program completed successfully!'
    except Exception as e:
        print(e)
        traceback.print_exc()
        run_result = 'Error: ' + str(e)
    end_time = time.time()
    print('Program ended at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    print('Execution time: ' + str(datetime.timedelta(seconds=end_time - start_time)))

    from snb_utils import Mailer

    subject = f'VSL020/{this_filename}'
    mail_content = f'Program ended.\n' + run_result
    Mailer.send_mail(subject, mail_content)
