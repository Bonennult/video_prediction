from .base_dataset import BaseVideoDataset
from .base_dataset import VideoDataset, SequenceExampleVideoDataset, VarLenFeatureVideoDataset
from .google_robot_dataset import GoogleRobotVideoDataset
from .sv2p_dataset import SV2PVideoDataset
from .softmotion_dataset import SoftmotionVideoDataset
from .kth_dataset import KTHVideoDataset
from .ucf101_dataset import UCF101VideoDataset
from .cartgripper_dataset import CartgripperVideoDataset


def get_dataset_class(dataset):
    dataset_mappings = {
        'google_robot': 'GoogleRobotVideoDataset',
        'sv2p': 'SV2PVideoDataset',
        'softmotion': 'SoftmotionVideoDataset',
        'bair': 'SoftmotionVideoDataset',  # alias of softmotion
        'kth': 'KTHVideoDataset',
        'ucf101': 'UCF101VideoDataset',
        'cartgripper': 'CartgripperVideoDataset',
    }
    dataset_class = dataset_mappings.get(dataset, dataset)   ### 第二个参数是default 5/3
    dataset_class = globals().get(dataset_class)   ### globals()，pyhton内置函数，返回全局变量的字典 5/3
                                    ### 使用get()是为了获得类的体，而不仅仅是类的名字（一个字符串） 5/3
    if dataset_class is None or not issubclass(dataset_class, BaseVideoDataset):
        raise ValueError('Invalid dataset %s' % dataset)
    return dataset_class
