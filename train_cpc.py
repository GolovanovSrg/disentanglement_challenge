import random
import json
import aicrowd_helpers
import numpy as np
import torch
import utils_pytorch as pyu

from sklearn.decomposition import PCA
from cpc import Encoder, PixelSNAIL, LinearPredictor, ReccurentPredictor, Trainer


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)


def read_config(path):
    with open(path, 'r') as file:
        config = json.load(file)
    return config


class PCAWrapper(torch.nn.Module):
    def __init__(self, sklearn_pca):
        super().__init__()

        if sklearn_pca.mean_ is not None:
            self.register_buffer('mean', torch.from_numpy(sklearn_pca.mean_))
        else:
            self.mean = None
        self.register_buffer('components', torch.from_numpy(sklearn_pca.components_.T))
        if sklearn_pca.whiten:
            self.register_buffer('explained_variance', torch.from_numpy(sklearn_pca.explained_variance_))
        else:
            self.explained_variance = None

    def forward(self, x):
        if self.mean is not None:
            x = x - self.mean
        x = torch.matmul(x, self.components)
        if self.explained_variance is not None:
            x = x / torch.sqrt(self.explained_variance)
        return x


class RepresentationExtractor(torch.nn.Module):
    def __init__(self, encoder, pca, batch_size=64):
        super().__init__()
        self.encoder = encoder
        self.pca = pca
        self.register_buffer('batch_size', torch.tensor(batch_size))

    def forward(self, x):
        result = []
        chunker = (x[pos:pos + self.batch_size.item()] for pos in range(0, len(x), self.batch_size.item()))
        for chunk in chunker:
            chunk = self.encoder(chunk, average=True)
            chunk = self.pca(chunk)
            result.append(chunk)
        return torch.cat(result, dim=0)


def main():
    seed = 0
    config_path = 'cpc_config.json'

    set_seed(seed)
    config = read_config(config_path)

    encoder_config = config['model']['encoder']
    encoder = Encoder(embedding_size=encoder_config['embedding_size'],
                      kernel_size=encoder_config['kernel_size'],
                      backbone_type=encoder_config['backbone_type'])

    autoregressor_config = config['model']['autoregressor']
    autoregressor = PixelSNAIL(in_channel=encoder_config['embedding_size'],
                               channel=autoregressor_config['channel'],
                               kernel_size=autoregressor_config['kernel_size'],
                               n_block=autoregressor_config['n_block'],
                               n_res_block=autoregressor_config['n_res_block'],
                               res_channel=autoregressor_config['res_channel'],
                               attention=autoregressor_config['attention'],
                               n_head=autoregressor_config['n_head'],
                               dropout=autoregressor_config['dropout'])
    
    predictor_config = config['model']['predictor']
    if predictor_config['reccurent']:
        predictor = ReccurentPredictor(in_channels=encoder_config['embedding_size'],
                                       out_channels=encoder_config['embedding_size'],
                                       n_predictions=predictor_config['n_predictions'])
    else:
        predictor = LinearPredictor(in_channels=encoder_config['embedding_size'],
                                    out_channels=encoder_config['embedding_size'],
                                    n_predictions=predictor_config['n_predictions'])

    trainer_config = config['trainer']
    trainer = Trainer(encoder, autoregressor, predictor,
                      optimizer_params=trainer_config['optimizer_params'],
                      devices=trainer_config['devices'],
                      n_jobs=trainer_config['n_jobs'])

    aicrowd_helpers.execution_start()
    aicrowd_helpers.register_progress(0.)

    train_dataset, test_dataset = pyu.get_datasets(test_part=trainer_config['test_part'])
    checkpoint_path = trainer_config['checkpoint_path']
    trainer.train(train_data=train_dataset, test_data=test_dataset, n_epochs=trainer_config['n_epochs'],
                  batch_size=trainer_config['batch_size'], best_checkpoint_path=checkpoint_path)

    print(f'Encoder weight from {checkpoint_path}')
    model_state = torch.load(checkpoint_path, map_location=trainer.device)
    trainer.encoder.module.load_state_dict(model_state['encoder'])

    pca_config = config['model']['pca']
    if pca_config['enable']:
        with torch.no_grad():
            pca = PCA(n_components=pca_config['n_components'], whiten=pca_config['whiten'], random_state=seed)
            pca_data = (train_dataset[i].unsqueeze(0).to(trainer.device) for i in range(pca_config['n_data']))
            pca_data = (trainer.encoder.module(d, average=True) for d in pca_data)
            pca_data = torch.cat([d.cpu() for d in pca_data], dim=0).numpy()
            pca.fit(pca_data)
            pca = PCAWrapper(pca)
    else:
        pca = torch.nn.Identity()

    aicrowd_helpers.register_progress(0.90)
    pyu.export_model(RepresentationExtractor(trainer.encoder.module, pca), input_shape=(1, 3, 64, 64))
    aicrowd_helpers.register_progress(1.0)


if __name__ == '__main__':
    main()
