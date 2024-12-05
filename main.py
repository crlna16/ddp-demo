import sys
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, Timer

from omegaconf import OmegaConf
from modules import Food101DataModule, Food101Model

import logging
log = logging.getLogger()

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
    log.info('Setting up the datamodule...')
    datamodule = Food101DataModule(batch_size=256)
    datamodule.prepare_data()
    datamodule.setup(stage='train')

    # setup the model
    log.info('Setting up the model...')
    model = Food101Model()

    # setup the callbacks
    log.info('Setting up the callbacks...')
    callbacks = [EarlyStopping(monitor='val/accuracy', mode='max', patience=3),
                 Timer()]

    # setup the trainer
    log.info('Setting up the trainer...')
    log.info('Trainer config', config['trainer'])
    trainer = L.Trainer(max_epochs=500,
                        logger=logger,
                        callbacks=callbacks,
                        **config['trainer']
                        )

    # fit
    log.info('Starting the fit')
    trainer.fit(model, datamodule)

    log.info('Timer callback report')
    for c in callbacks :
        try:
            log.info(f'Training took {c.time_elapsed():.0f} seconds')
            log.info(f'Validation took {c.time_elapsed("validate"):.0f} seconds')
        except:
            pass

    # test
    test_datamodule = Food101DataModule(batch_size=512)
    test_datamodule.prepare_data(split='test')
    test_datamodule.setup(stage='train')

    test_trainer = L.Trainer(devices=1, num_nodes=1)

    test_trainer.test(model, datamodule=test_datamodule)




if __name__=='__main__':
    main(sys.argv[1])
