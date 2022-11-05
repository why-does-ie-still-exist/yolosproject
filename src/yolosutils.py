import torchvision
import os
from transformers import AutoFeatureExtractor, AutoModelForObjectDetection
from pytorch_lightning import LightningModule
from torch.optim import AdamW

class CocoDetection(torchvision.datasets.CocoDetection):
    """
    This class is for constructing our dataset, which we load from the roboflow dataset
    """
    def __init__(self, img_folder, feature_extractor, train=True):
        ann_file = os.path.join(img_folder, "_annotations.coco.json")  # json annotations are here
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        img, target = super(CocoDetection, self).__getitem__(idx)

        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.feature_extractor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()  # remove batch dimension
        target = encoding["labels"][0]  # remove batch dimension

        return pixel_values, target

MODEL_PREFIX = 'hustvl/yolos-'

def get_yolos_feature_extractor(modelsize):
    return AutoFeatureExtractor.from_pretrained(MODEL_PREFIX+modelsize, size=512, max_size=864)

def get_collate_fn(feature_extractor):
    def collate_fn(batch):
        pixel_values = [item[0] for item in batch]  # these are the image tensors, they are the first index in each item
        encoding = feature_extractor.pad(pixel_values,
                                         return_tensors="pt")  # we encode our image before doing a forward pass
        labels = [item[1] for item in batch]
        batch = {}
        batch['pixel_values'] = encoding['pixel_values']  # take note here, this is what a batch looks like
        batch['labels'] = labels
        return batch
    return collate_fn

class YoloS(LightningModule):

    def __init__(self, lr, weight_decay, batch_size, model_size, num_labels, val_dataloader_in, train_dataloader_in=None):
        super().__init__()
        # replace COCO classification head with custom head
        self.model = AutoModelForObjectDetection.from_pretrained(MODEL_PREFIX + model_size,
                                                                 # num_labels=num_labels,
                                                                 ignore_mismatched_sizes=True)
        # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()  # adding this will save the hyperparameters to W&B too
        self.stored_val_dataloader = val_dataloader_in
        self.stored_train_dataloader = train_dataloader_in
        self.batch_size = batch_size

    def on_train_start(self):
        self.log("batch_size", float(self.batch_size))

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        return outputs

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("train/loss",
                 loss)  # logging metrics with a forward slash will ensure the train and validation metrics as split into 2 separate sections in the W&B workspace
        for k, v in loss_dict.items():
            self.log("train/" + k,
                     v.item())  # logging metrics with a forward slash will ensure the train and validation metrics as split into 2 separate sections in the W&B workspace

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation/loss",
                 loss)  # logging metrics with a forward slash will ensure the train and validation metrics as split into 2 separate sections in the W&B workspace
        for k, v in loss_dict.items():
            self.log("validation/" + k,
                     v.item())  # logging metrics with a forward slash will ensure the train and validation metrics as split into 2 separate sections in the W&B workspace

        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr,
                                      weight_decay=self.weight_decay)
        return optimizer

    def val_dataloader(self):
        return self.stored_val_dataloader

    def train_dataloader(self):
        return self.stored_train_dataloader