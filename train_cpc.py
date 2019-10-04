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


def main():
    seed = 0
    config_path = 'cpc_config.json'

    set_seed(seed)
    config = read_config(config_path)

    encoder_config = config['model']['encoder']
    encoder = Encoder(embedding_size=encoder_config['embedding_size'],
                      kernel_size=encoder_config['kernel_size'],
                      pretrained=encoder_config['pretrained'],
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

    model_state = torch.load(checkpoint_path, map_location='cpu')
    representator_state = {'config': config['model'],
                           'encoder': model_state['encoder'],
                           'autoregressor': model_state['autoregressor']}
    pyu.export_model(representator_state)

    aicrowd_helpers.register_progress(1.0)


if __name__ == '__main__':
    main()
