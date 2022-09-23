# Shree KRISHNAYa Namaha
# Loads RealEstate Data for NeRF
# Author: Nagabhushan S N
# Last Modified: 23/09/2022

from pathlib import Path
from typing import Optional

import numpy
import pandas
import skimage.io
from data_loaders.DataLoaderParent import DataLoaderParent


class RealEstateDataLoader(DataLoaderParent):
    def __init__(self, configs: dict, data_dirpath: Path, mode: Optional[str]):
        super(RealEstateDataLoader, self).__init__()
        self.configs = configs
        self.data_dirpath = data_dirpath
        self.mode = mode
        self.scene_num = int(configs['data_loader']['scene_id'])
        return

    def load_data(self):
        frame_nums = self.get_frame_nums()
        data_dict = {
            'frame_nums': frame_nums,
        }

        data_dict['nerf_data'] = self.load_nerf_data(data_dict)

        return data_dict

    def get_frame_nums(self):
        set_num = self.configs['data_loader']['train_set_num']
        video_datapath = self.data_dirpath / f'TrainTestSets/Set{set_num:02}/{self.mode.capitalize()}VideosData.csv'
        video_data = pandas.read_csv(video_datapath)
        frame_nums = video_data.loc[video_data['scene_num'] == self.scene_num]['pred_frame_num'].to_numpy()
        return frame_nums

    def load_nerf_data(self, data_dict):
        frame_nums = data_dict['frame_nums']
        images_dirpath = self.data_dirpath / f'test/DatabaseData/{self.scene_num:05}/rgb'
        if not images_dirpath.exists():
            print(f'{images_dirpath.as_posix()} does not exist, returning.')
            return
        images_paths = [images_dirpath / f'{frame_num:04}.png' for frame_num in frame_nums]
        images = [self.read_image(image_path) for image_path in images_paths]
        images = numpy.stack(images)

        bounds = numpy.array([1, 100]).astype('float32')

        extrinsics_path = self.data_dirpath / f'test/DatabaseData/{self.scene_num:05}/CameraExtrinsics.csv'
        extrinsic_matrices = numpy.loadtxt(extrinsics_path.as_posix(), delimiter=',').reshape((-1, 4, 4))
        poses = extrinsic_matrices[frame_nums]

        intrinsics_path = self.data_dirpath / f'test/DatabaseData/{self.scene_num:05}/CameraIntrinsics.csv'
        intrinsic_matrices = numpy.loadtxt(intrinsics_path.as_posix(), delimiter=',').reshape((-1, 3, 3))
        intrinsics = intrinsic_matrices[frame_nums]
        # assert all intrinsics are equal
        for intrinsic in intrinsics:
            assert numpy.allclose(intrinsics[0], intrinsic)

        fx, fy = intrinsics[0][0, 0], intrinsics[0][1, 1]
        h, w = images.shape[1:3]

        return_dict = {
            'images': images,
            'poses': poses,
            'resolution': (h, w),
            'focal_length': (fx, fy),
            'bounds': bounds,
        }
        return return_dict

    @staticmethod
    def read_image(path: Path, mmap_mode: str = None):
        if path.suffix in ['.png']:
            image = skimage.io.imread(path.as_posix())
        elif path.suffix == '.npy':
            image = numpy.load(path.as_posix(), mmap_mode=mmap_mode)
        else:
            raise RuntimeError(f'Unknown image format: {path.as_posix()}')
        return image
