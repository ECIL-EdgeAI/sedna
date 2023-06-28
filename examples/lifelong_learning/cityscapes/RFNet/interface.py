import os

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToPILImage
from sklearn.model_selection import train_test_split
from sedna.common.config import Context
from sedna.datasources import BaseDataSource
from sedna.common.file_ops import FileOps
from sedna.common.log import LOGGER

from dataloaders import custom_transforms as tr
from dataloaders.utils import Colorize
from utils.args import TrainingArguments, EvaluationArguments
from estimators.train import Trainer
from estimators.eval import Validator, load_my_state_dict
from accuracy import accuracy

os.environ["BACKEND_TYPE"] = ''

classes = ["road", "sidewalk", "building", "wall", "fence", "pole", "light",
           "sign", "vegetation", "terrain", "sky", "pedestrian", "rider",
           "car", "truck", "bus", "train", "motorcycle", "bicycle", "stair",
           "curb", "ramp", "runway", "flowerbed", "door", "CCTV camera",
           "Manhole", "hydrant", "belt", "dustbin", "ignore"]


def preprocess_url(image_urls):
    transformed_images = []
    for paths in image_urls:
        if len(paths) == 2:
            img_path, depth_path = paths
            _img = Image.open(img_path).convert(
                'RGB').resize((2048, 1024), Image.BILINEAR)
            _depth = Image.open(depth_path).resize(
                (2048, 1024), Image.BILINEAR)
        else:
            img_path = paths[0]
            _img = Image.open(img_path).convert(
                'RGB').resize((2048, 1024), Image.BILINEAR)
            _depth = _img

        sample = {'image': _img, 'depth': _depth, 'label': _img}
        composed_transforms = transforms.Compose([
            # tr.CropBlackArea(),
            # tr.FixedResize(size=self.args.crop_size),
            tr.Normalize(
                mean=(
                    0.485, 0.456, 0.406), std=(
                    0.229, 0.224, 0.225)),
            tr.ToTensor()])

        transformed_images.append((composed_transforms(sample), img_path))

    return transformed_images


def preprocess_frames(frames):
    composed_transforms = transforms.Compose([
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr.ToTensor()])

    trainsformed_frames = []
    for frame in frames:
        img = frame.get('image')
        img = cv2.resize(np.array(img), (2048, 1024),
                         interpolation=cv2.INTER_CUBIC)
        img = Image.fromarray(np.array(img))
        sample = {'image': img, "depth": img, "label": img}
        trainsformed_frames.append((composed_transforms(sample), ""))

    return trainsformed_frames


class Estimator:
    def __init__(self, **kwargs):
        self.cuda = torch.cuda.is_available()
        kwargs.update(Context.parameters)
        
        self.classes = kwargs.get("classes", classes)
        self.train_args = TrainingArguments(**kwargs)
        self.val_args = EvaluationArguments(**kwargs)

        self.train_args.resume = Context.get_parameters(
            "PRETRAINED_MODEL_URL", None)
        self.trainer = None
        self.train_model_url = None

        self.validator = Validator(self.val_args)

    def train(self, train_data, valid_data=None, **kwargs):
        if not valid_data:
            train_ratio = float(Context.get_parameters("train_ratio", 0.9))
            valid_data = BaseDataSource(data_type="eval")
            train_data.x, valid_data.x, train_data.y, valid_data.y = \
                train_test_split(train_data.x, train_data.y,
                                 train_size=train_ratio)

        self.trainer = Trainer(
            self.train_args, train_data=train_data, valid_data=valid_data)
        LOGGER.info("Total epoches: {}".format(self.trainer.args.epochs))
        for epoch in range(
                self.trainer.args.start_epoch,
                self.trainer.args.epochs):
            # if epoch == 0 and self.trainer.val_loader:
            #     self.trainer.validation(epoch)
            self.trainer.training(epoch)

            if self.trainer.args.no_val and (epoch %
                                             self.trainer.args.eval_interval == (
                                                 self.trainer.args.eval_interval -
                                                 1) or epoch == self.trainer.args.epochs - 1):
                # save checkpoint when it meets eval_interval or the training finishes
                is_best = False
                train_model_url = self.trainer.saver.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.trainer.model.state_dict(),
                    'optimizer': self.trainer.optimizer.state_dict(),
                    'best_pred': self.trainer.best_pred,
                }, is_best)

            # if not self.trainer.args.no_val and \
            #         epoch % self.train_args.eval_interval == (self.train_args.eval_interval - 1) \
            #         and self.trainer.val_loader:
            #     self.trainer.validation(epoch)

        self.trainer.writer.close()
        self.train_model_url = train_model_url

        return {"mIoU": self.trainer.validation(epoch)}

    def predict(self, data, **kwargs):
        if isinstance(data[0], dict):
            data = preprocess_frames(data)

        if isinstance(data[0], np.ndarray):
            data = preprocess_url(data)

        self.validator.test_loader = DataLoader(
            data,
            batch_size=self.val_args.test_batch_size,
            shuffle=False,
            pin_memory=False)

        return self.validator.validate()

    def evaluate(self, data, **kwargs):
        predictions = self.predict(data.x)
        return accuracy(data.y, predictions)

    def load(self, model_url, **kwargs):
        if model_url:
            LOGGER.info(f"Load model from {model_url}")
            self.validator.new_state_dict = \
                torch.load(model_url,
                           map_location="cpu" if not self.cuda else None)
            self.validator.model = load_my_state_dict(
                self.validator.model,
                self.validator.new_state_dict['state_dict'],
                self.cuda)

            self.train_args.resume = model_url
        else:
            raise Exception("Model url does not exist.")

    def save(self, model_path=None):
        if not model_path:
            LOGGER.warning(f"Not specify model path.")
            return self.train_model_url

        return FileOps.upload(self.train_model_url, model_path)


def save_predicted_image(img_url, image, pred, image_name):
    '''
    Sample post processing function invoked by Sedna 
    to upload the inference results
    '''

    merge_label_name = os.path.join(img_url, f"merge_{image_name}")
    color_label_name = os.path.join(img_url, f"color_{image_name}")
    label_name = os.path.join(img_url, f"label_{image_name}")
    os.makedirs(os.path.dirname(merge_label_name), exist_ok=True)
    os.makedirs(os.path.dirname(color_label_name), exist_ok=True)
    os.makedirs(os.path.dirname(label_name), exist_ok=True)

    # Save prediction images
    pred = torch.from_numpy(pred).byte()
    pre_color = Colorize()(pred)
    pre_label = pred

    pre_color_image = ToPILImage()(pre_color[0])
    image_merge(image, pre_color_image, merge_label_name)
    pre_color_image.save(color_label_name)
    pre_label_image = ToPILImage()(pre_label)
    pre_label_image.save(label_name)

    return (merge_label_name, color_label_name, label_name)


def image_merge(image, label, save_name):
    '''
    Merge original image and predicted image into one image
    '''
    image = image.resize(label.size, Image.BILINEAR)
    image = image.convert('RGBA')
    label = label.convert('RGBA')
    image = Image.blend(image, label, 0.6).resize(image.size)
    image.save(save_name)