# Shree KRISHNAya Namaha
# Common Tester for different models and datasets
# Author: Nagabhushan S N
# Last Modified: 23/09/2022

import json
from pathlib import Path

import numpy
import simplejson
import skimage.io
import torch
from deepdiff import DeepDiff
from tqdm import tqdm

from data_preprocessors.DataPreprocessorFactory import get_data_preprocessor
from models.ModelFactory import get_model
from utils import CommonUtils01 as CommonUtils

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class NerfTester:
    def __init__(self, train_configs: dict, model_configs: dict, test_configs: dict, root_dirpath: Path, project_dirpath: Path):
        self.train_configs = train_configs
        self.test_configs = test_configs
        self.root_dirpath = root_dirpath
        self.project_dirpath = project_dirpath
        self.database_dirpath = self.project_dirpath / 'Databases' / self.test_configs['database_dirpath']
        self.data_preprocessor = None
        self.model = None
        self.model_configs = model_configs
        self.device = CommonUtils.get_device(test_configs['device'])

        self.build_model()
        return

    def build_model(self):
        self.data_preprocessor = get_data_preprocessor(self.train_configs, mode='test', model_configs=self.model_configs)
        self.model = get_model(self.train_configs, self.model_configs)
        return

    def load_model(self, model_path: Path):
        checkpoint_state = torch.load(model_path, map_location=self.device)
        iter_num = checkpoint_state['iteration_num']
        self.model.load_state_dict(checkpoint_state['model_state_dict'])
        self.model.eval()

        train_dirname = model_path.parent.parent.parent.stem
        scene_dirname = model_path.parent.parent.stem
        model_name = model_path.stem
        print(f'Loaded Model in {train_dirname}/{scene_dirname}/{model_name} trained for {iter_num} iterations')
        return

    def predict_frame(self, camera_pose: numpy.ndarray, view_camera_pose: numpy.ndarray = None):
        input_dict = self.data_preprocessor.create_test_data(camera_pose, view_camera_pose)

        with torch.no_grad():
            output_batch = self.model(input_dict)

        processed_output = self.data_preprocessor.retrieve_inference_outputs(output_batch)
        return processed_output
    
    @staticmethod
    def save_image(path: Path, image: numpy.ndarray):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == '.png':
            skimage.io.imsave(path.as_posix(), image)
        elif path.suffix == '.npy':
            numpy.save(path.as_posix(), image)
        else:
            raise RuntimeError(f'Unknown image format: {path.as_posix()}')
        return

    @staticmethod
    def save_depth(path: Path, depth: numpy.ndarray, as_png: bool = False):
        path.parent.mkdir(parents=True, exist_ok=True)
        depth_image = numpy.round(depth / depth.max() * 255).astype('uint8')
        if path.suffix == '.png':
            skimage.io.imsave(path.as_posix(), depth_image, check_contrast=False)
        elif path.suffix == '.npy':
            numpy.save(path.as_posix(), depth)
            if as_png:
                png_path = path.parent / f'{path.stem}.png'
                skimage.io.imsave(png_path.as_posix(), depth_image, check_contrast=False)
        else:
            raise RuntimeError(f'Unknown depth format: {path.as_posix()}')
        return


def save_configs(output_dirpath: Path, configs: dict):
    configs_path = output_dirpath / 'Configs.json'
    if configs_path.exists():
        with open(configs_path.as_posix(), 'r') as configs_file:
            old_configs = json.load(configs_file)
        for key in old_configs.keys():
            if key not in configs.keys():
                configs[key] = old_configs[key]
        if configs != old_configs:
            raise RuntimeError(f'Configs mismatch while resuming testing: {DeepDiff(old_configs, configs)}')
    with open(configs_path.as_posix(), 'w') as configs_file:
        simplejson.dump(configs, configs_file, indent=4)
    return


def start_testing(test_configs: dict, scenes_data: dict, output_dir_suffix: str = '', save_depth: bool = False, 
                  save_depth_var: bool = False):
    device = CommonUtils.get_device(test_configs['device'])
    if device.type == 'cuda':
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    elif device.type == 'cpu':
        torch.set_default_tensor_type('torch.FloatTensor')
    else:
        raise RuntimeError(f'Unknown device type: {device.type}')

    root_dirpath = Path('../')
    project_dirpath = root_dirpath / '../../../../'
    database_dirpath = project_dirpath / 'Databases' / test_configs['database_dirpath']
    output_dirpath = root_dirpath / f"Runs/Testing/Test{test_configs['test_num']:04}"

    train_num = test_configs['train_num']
    model_name = test_configs['model_name']
    train_dirpath = root_dirpath / f'Runs/Training/Train{train_num:04}'
    train_configs_path = train_dirpath / 'Configs.json'
    if not train_configs_path.exists():
        print(f'Train Configs does not exist at {train_configs_path.as_posix()}. Skipping.')
        return
    with open(train_configs_path.as_posix(), 'r') as configs_file:
        train_configs = simplejson.load(configs_file)

    for scene_id in scenes_data:
        scene_data = scenes_data[scene_id]
        train_configs['data_loader']['scene_id'] = scene_id

        trained_model_configs_path = train_dirpath / f'{scene_id}/ModelConfigs.json'
        if not trained_model_configs_path.exists():
            print(f'Scene {scene_id}: Trained Model Configs does not exist at {trained_model_configs_path.as_posix()}. Skipping.')
            continue
        with open(trained_model_configs_path.as_posix(), 'r') as configs_file:
            trained_model_configs = simplejson.load(configs_file)
        model_path = train_dirpath / f"{scene_id}/SavedModels/{model_name}"
        if not model_path.exists():
            print(f'Scene {scene_id}: Model does not exist at {model_path.as_posix()}. Skipping.')
            continue

        # Build the model
        tester = NerfTester(train_configs, trained_model_configs, test_configs, root_dirpath, project_dirpath)
        tester.load_model(model_path)

        # Test and save
        scene_output_dirname = scene_data['output_dirname']
        scene_output_dirpath = output_dirpath / f'{scene_output_dirname}{output_dir_suffix}'
        frame_nums = scene_data['frames_data'].keys()
        for frame_num in tqdm(frame_nums, desc=f'{scene_id}'):
            frame_data = scene_data['frames_data'][frame_num]
            frame_output_path = scene_output_dirpath / f'PredictedFrames/{frame_num:04}.png'
            depth_output_path = scene_output_dirpath / f'PredictedDepths/{frame_num:04}.npy'
            depth_var_output_path = scene_output_dirpath / f'PredictedDepthsVariance/{frame_num:04}.npy'
            depth_ndc_output_path = scene_output_dirpath / f'PredictedDepths/{frame_num:04}_ndc.npy'
            depth_var_ndc_output_path = scene_output_dirpath / f'PredictedDepthsVariance/{frame_num:04}_ndc.npy'

            inference_required = not frame_output_path.exists()
            if save_depth:
                inference_required = inference_required or (not depth_output_path.exists())
            if save_depth_var:
                inference_required = inference_required or (not depth_var_output_path.exists())
            if inference_required:
                tgt_pose = frame_data['extrinsic']
                view_tgt_pose = frame_data['extrinsic_viewcam'] if 'extrinsic_viewcam' in frame_data else None
                predictions = tester.predict_frame(tgt_pose, view_tgt_pose)

                tester.save_image(frame_output_path, predictions['image'])
                if save_depth:
                    tester.save_depth(depth_output_path, predictions['depth'], as_png=True)
                    if 'depth_ndc' in predictions:
                        tester.save_depth(depth_ndc_output_path, predictions['depth_ndc'], as_png=True)
                if save_depth_var:
                    tester.save_depth(depth_var_output_path, predictions['depth_var'], as_png=True)
                    if 'depth_var_ndc' in predictions:
                        tester.save_depth(depth_var_ndc_output_path, predictions['depth_var_ndc'], as_png=True)
    return output_dirpath
