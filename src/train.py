import argparse

from roboflow import Roboflow

from yolosutils import CocoDetection, get_collate_fn, YoloS, get_yolos_feature_extractor

parser = argparse.ArgumentParser(
    prog='YOLOS Trainer',
    description='Trains YOLOS on https://universe.roboflow.com/rm-2021/armor-plates')
parser.add_argument('--size', required=True, help='model size, can be "tiny", "small", or "base"',
                    choices=['tiny', 'small', 'base'])
parser.add_argument('--batch_size', type=int, default=20, help='batch size for training')
parser.add_argument('--patience', type=int, default=5, help='patience(num epochs) for early stopping')
parser.add_argument('--min_delta', type=float, default=0.03, help='minimum delta(in val loss) for early stopping')
args = parser.parse_args()
MODEL_SIZE = args.size
BATCH_SIZE = args.batch_size
PATIENCE = args.patience
MIN_DELTA = args.min_delta

ROBOFLOW_API_KEY = "6wzCHB3PAi6vCKlHrZT5"
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
# https://universe.roboflow.com/rm-2021/armor-plates
project = rf.workspace("rm-2021").project("armor-plates")
dataset = project.version(26).download("coco")

feature_extractor = get_yolos_feature_extractor(MODEL_SIZE)

train_dataset = CocoDetection(img_folder=(dataset.location + '/train'), feature_extractor=feature_extractor)
val_dataset = CocoDetection(img_folder=(dataset.location + '/valid'), feature_extractor=feature_extractor, train=False)

from torch.utils.data import DataLoader

collate_fn = get_collate_fn(feature_extractor)

train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, shuffle=True, batch_size=BATCH_SIZE, num_workers=0)
val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE, num_workers=0)

model = YoloS(lr=2.5e-5, weight_decay=1e-4, batch_size=BATCH_SIZE, model_size=MODEL_SIZE, num_labels=2,
              val_dataloader_in=val_dataloader, train_dataloader_in=train_dataloader)

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

callbacks = [EarlyStopping(monitor="validation/loss", patience=PATIENCE, mode="min", min_delta=MIN_DELTA),
             ModelCheckpoint(monitor="validation/loss", mode="min")]

trainer = Trainer(max_epochs=50, gradient_clip_val=0.1, callbacks=callbacks,  # auto_scale_batch_size='binsearch',
                  devices=-1, accelerator='gpu', num_sanity_val_steps=0)
trainer.tune(model)
trainer.fit(model)