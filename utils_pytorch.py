from copy import deepcopy
import os
from collections import namedtuple

import numpy as np
import torch
from albumentations import Compose, HorizontalFlip, VerticalFlip, RandomRotate90, RandomSizedCrop

# ------ Data Loading ------
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

import os
if 'DISENTANGLEMENT_LIB_DATA' not in os.environ:
    os.environ.update({'DISENTANGLEMENT_LIB_DATA': os.path.join(os.path.dirname(__file__),
                                                                'scratch',
                                                                'dataset')})
# noinspection PyUnresolvedReferences
from disentanglement_lib.data.ground_truth.named_data import get_named_ground_truth_data
# --------------------------


ExperimentConfig = namedtuple('ExperimentConfig',
                              ('base_path', 'experiment_name', 'dataset_name'))


def get_config():
    """
    This function reads the environment variables AICROWD_OUTPUT_PATH,
    AICROWD_EVALUATION_NAME and AICROWD_DATASET_NAME and returns a
    named tuple.
    """
    return ExperimentConfig(base_path=os.getenv("AICROWD_OUTPUT_PATH", "./scratch/shared"),
                            experiment_name=os.getenv("AICROWD_EVALUATION_NAME", "experiment_name"),
                            dataset_name=os.getenv("AICROWD_DATASET_NAME", "cars3d"))


def get_dataset_name():
    """Reads the name of the dataset from the environment variable `AICROWD_DATASET_NAME`."""
    return os.getenv("AICROWD_DATASET_NAME", "cars3d")


def use_cuda():
    """
    Whether to use CUDA for evaluation. Returns True if CUDA is available and
    the environment variable AICROWD_CUDA is not set to False.
    """
    return torch.cuda.is_available() and os.getenv('AICROWD_CUDA', True)


def get_model_path(base_path=None, experiment_name=None, make=True):
    """
    This function gets the path to where the model is expected to be stored.

    Parameters
    ----------
    base_path : str
        Path to the directory where the experiments are to be stored.
        This defaults to AICROWD_OUTPUT_PATH (see `get_config` above) and which in turn
        defaults to './scratch/shared'.
    experiment_name : str
        Name of the experiment. This defaults to AICROWD_EVALUATION_NAME which in turn
        defaults to 'experiment_name'.
    make : Makes the directory where the returned path leads to (if it doesn't exist already)

    Returns
    -------
    str
        Path to where the model should be stored (to be found by the evaluation function later).
    """
    base_path = os.getenv("AICROWD_OUTPUT_PATH","../scratch/shared") \
        if base_path is None else base_path
    experiment_name = os.getenv("AICROWD_EVALUATION_NAME", "experiment_name") \
        if experiment_name is None else experiment_name
    model_path = os.path.join(base_path, experiment_name, 'representation', 'pytorch_model.pt')
    if make:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.join(os.path.dirname(model_path), 'results'), exist_ok=True)
    return model_path


def export_model(model_state, path=None):
    path = get_model_path() if path is None else path
    torch.save(model_state, path)


def import_model(path=None):
    from cpc.models import Encoder, RepresentationExtractor
    from cpc.pixelsnail import PixelSNAIL

    path = get_model_path() if path is None else path
    model_state = torch.load(path, map_location='cpu')

    config = model_state['config']

    encoder_config = config['encoder']
    encoder = Encoder(embedding_size=encoder_config['embedding_size'],
                      kernel_size=encoder_config['kernel_size'],
                      backbone_type=encoder_config['backbone_type'])
    encoder.load_state_dict(model_state['encoder'])

    autoregressor_config = config['autoregressor']
    autoregressor = PixelSNAIL(in_channel=encoder_config['embedding_size'],
                               channel=autoregressor_config['channel'],
                               kernel_size=autoregressor_config['kernel_size'],
                               n_block=autoregressor_config['n_block'],
                               n_res_block=autoregressor_config['n_res_block'],
                               res_channel=autoregressor_config['res_channel'],
                               attention=autoregressor_config['attention'],
                               n_head=autoregressor_config['n_head'],
                               dropout=autoregressor_config['dropout'])
    autoregressor.load_state_dict(model_state['autoregressor'])

    representator = RepresentationExtractor(encoder, autoregressor, batch_size=64)

    return representator


def make_representor(model, cuda=None):
    cuda = use_cuda() if cuda is None else cuda
    model = model.cuda() if cuda else model.cpu()
    model.eval()

    # Define the representation function
    def _represent(x):
        assert isinstance(x, np.ndarray), \
            "Input to the representation function must be a ndarray."
        assert x.ndim == 4, \
            "Input to the representation function must be a four dimensional NHWC tensor."
        # Convert from NHWC to NCHW
        x = np.moveaxis(x, 3, 1)
        # Convert to torch tensor and evaluate
        x = torch.from_numpy(x).float().to('cuda' if cuda else 'cpu')
        with torch.no_grad():
            y = model(x)
        y = y.cpu().numpy()
        assert y.ndim == 2, \
            "The returned output from the representor must be two dimensional (NC)."
        return y

    return _represent


class TrainTransform:
    def __init__(self):
        self.transforms = Compose([HorizontalFlip(),
                                   VerticalFlip(),
                                   RandomRotate90(),
                                   RandomSizedCrop(min_max_height=(56, 64),
                                                   height=64, width=64)])

    def __call__(self, image):
        return self.transforms(image=image)['image']


class DLIBDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        output = self.images[idx].astype(np.float32)
        if self.transform is not None:
            output = self.transform(output)

        # Convert output to CHW from HWC
        output = np.moveaxis(output, 2, 0)
        return torch.from_numpy(output)


def get_datasets(name=None, test_part=0.2):
    name = get_dataset_name() if name is None else name
    data = get_named_ground_truth_data(name).images

    idx_perm = np.random.permutation(len(data))
    n_test_data = int(len(idx_perm) * test_part)
    test_idxs, train_idxs = idx_perm[:n_test_data], idx_perm[n_test_data:]
    test_data = [data[idx] for idx in test_idxs]
    train_data = [data[idx] for idx in train_idxs]
    train_transform = TrainTransform()

    return DLIBDataset(train_data, transform=train_transform), DLIBDataset(test_data)
