import pytorch_lightning as pl
from model_pl import YOLOv9LightningModule
from datasets_pl import CustomDataModule
from arguments import training_arguments

opt = training_arguments(True)

if __name__ == '__main__':

    model = YOLOv9LightningModule()
    data_module = CustomDataModule(
        data_yaml_path=opt.data,
        opt = opt,
        hyp_yaml_path=opt.hyp
    )

    trainer = pl.Trainer(devices=1, accelerator="gpu", max_epochs=opt.epochs)

    trainer.fit(model, data_module)
