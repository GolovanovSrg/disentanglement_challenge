import random
import json
import aicrowd_helpers
import torch
import utils_pytorch as pyu

from cpc import Encoder, PixelSNAIL, Predictor, Trainer


def set_seed(seed=0):
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)


def read_config(path):
    with open(path, 'r') as file:
        config = json.load(file)
    return config


class RepresentationExtractor(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.encoder.eval()

    def forward(self, x):
        with torch.no_grad():
            return self.encoder(x, average=True)


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
    predictor = Predictor(in_channels=encoder_config['embedding_size'],
                          out_channels=encoder_config['embedding_size'],
                          n_predictions=predictor_config['n_predictions'])

    trainer_config = config['trainer']
    trainer = Trainer(encoder, autoregressor, predictor,
                      optimizer_params=trainer_config['optimizer_params'],
                      device=trainer_config['device'],
                      n_jobs=trainer_config['n_jobs'])

    aicrowd_helpers.execution_start()
    aicrowd_helpers.register_progress(0.)

    dataset = pyu.get_dataset(seed=seed, iterator_len=100000)
    trainer.train(train_data=dataset, n_epochs=trainer_config['n_epochs'], batch_size=trainer_config['batch_size'])

    aicrowd_helpers.register_progress(0.90)
    pyu.export_model(RepresentationExtractor(trainer.encoder), input_shape=(1, 3, 64, 64))
    aicrowd_helpers.register_progress(1.0)


if __name__ == '__main__':
    main()
