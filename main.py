import sys
import time
import numpy as np
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.utilities import rank_zero_only
from sklearn.metrics import accuracy_score

from omegaconf import OmegaConf
from modules import Food101DataModule, Food101Model

import logging
log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler(sys.stdout))

for level in ( "debug", "info", "warning", "error", "exception", "fatal", "critical",):
        setattr(log, level, rank_zero_only(getattr(log, level)))

def main(config_file):
    config = OmegaConf.load(config_file)

    # log runs with MLFlow
    logger = L.pytorch.loggers.MLFlowLogger(
        experiment_name=config['name'],
        tracking_uri='./mlflow-logs',
        log_model=True
    )
    logger.log_hyperparams(locals())

    # setup the datamodule
    print('Setting up the datamodule...')
    datamodule = Food101DataModule(batch_size=256)
    datamodule.prepare_data()
    datamodule.setup(stage='train')

    # setup the model
    print('Setting up the model...')
    model = Food101Model()

    # setup the callbacks
    print('Setting up the callbacks...')
    callbacks = [EarlyStopping(monitor='val/accuracy', mode='max', patience=3)]

    # setup the trainer
    print('Setting up the trainer...')
    print('Trainer config', config['trainer'])
    trainer = L.Trainer(logger=logger,
                        callbacks=callbacks,
                        **config['trainer']
                        )

    # fit
    print('Starting the fit')
    t0 = time.time()
    trainer.fit(model, datamodule)
    print(f'Model completed {model.current_epoch - 1} epochs in {time.time() - t0:.2f} seconds')

    # test
    print('Entering test stage...')
    test_datamodule = Food101DataModule(batch_size=12)
    test_datamodule.prepare_data(split='test')
    test_datamodule.setup(stage='test')
    
    # quick manual loop
    test_dataloader = test_datamodule.test_dataloader()

    all_y = []
    all_yhat = []

    for i, batch in enumerate(test_dataloader):
        x, y = batch
        with torch.no_grad():
            logits = model(x)
            yhat = torch.argmax(torch.softmax(logits, dim=1), dim=1)

        all_y.append(y.cpu().numpy())
        all_yhat.append(yhat.cpu().numpy())

    print('Finished test set predictions')

    all_y = np.concatenate(all_y)
    all_yhat = np.concatenate(all_yhat)

    print('Calculating multiclass accuracy...')

    macc = accuracy_score(all_y, all_yhat)

    print(f'Mean accuracy: {macc:.4f}')



if __name__=='__main__':
    main(sys.argv[1])
