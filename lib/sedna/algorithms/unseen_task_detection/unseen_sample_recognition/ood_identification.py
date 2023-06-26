import os
from typing import Tuple, List

import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from PIL import Image
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sedna.common.constant import KBResourceConstant
from sedna.backend import set_backend
from sedna.common.file_ops import FileOps
from sedna.datasources import BaseDataSource
from sedna.common.config import BaseConfig
from sedna.common.log import LOGGER
from sedna.common.class_factory import ClassFactory, ClassType
from sedna.algorithms.seen_task_learning.artifact import Model, Task

__all__ = ('OodIdentification',)


@ClassFactory.register(ClassType.UTD)
class OodIdentification:
    """
    Corresponding to `OodIdentification`

    Parameters
    ----------
    task_extractor : Dict
        used to match target tasks
    origins: List[Metadata]
        metadata is usually a class feature
        label with a finite values.
    """

    def __init__(self, task_index, **kwargs):
        if isinstance(task_index, str):
            task_index = FileOps.load(task_index)

        self.cuda = torch.cuda.is_available()
        self.seen_task_key = KBResourceConstant.SEEN_TASK.value
        self.task_group_key = KBResourceConstant.TASK_GROUPS.value
        self.extractor_key = KBResourceConstant.EXTRACTOR.value

        self.base_model = kwargs.get("base_model") # (num_class=31)
        self.backup_model = kwargs.get('OOD_backup_model')
        if not self.backup_model:
            self.seen_extractor = task_index.get(
                self.seen_task_key).get(self.extractor_key)
            if isinstance(self.seen_extractor, str):
                self.seen_extractor = FileOps.load(self.seen_extractor)
            self.seen_task_groups = task_index.get(
                self.seen_task_key).get(self.task_group_key)
            self.seen_models = [task.model for task in self.seen_task_groups]
        else:
            self.backup_model = FileOps.download(self.backup_model)

        self.OOD_thresh = float(kwargs.get("OOD_thresh", 1.0))
        self.preprocess_func = kwargs.get("preprocess_func")
        self.glcm_thresh = float(kwargs.get("glcm_thresh", 0.9445))
        self.sample_transform = lambda x, size: F.interpolate(
            x, size, mode='nearest')
        self.softmax = torch.nn.Softmax(
            dim=self.base_model.train_args.num_class)

        self.meta_model_stage = kwargs.get("stage", "inference")
        if self.meta_model_stage.lower() == "inference":
            ood_model_url = kwargs.get("OOD_model")
            if not ood_model_url:
                ood_model_url = task_index.get("meta_estimators").get(
                    "unseen_sample_recognition_estimator")
            self.ood_model = FileOps.load(ood_model_url)

        self.origin = kwargs.get("origin", "garden")

    def __call__(self, samples: BaseDataSource, **
                 kwargs) -> Tuple[BaseDataSource, BaseDataSource]:
        '''
        Parameters
        ----------
        samples : BaseDataSource
            inference samples

        Returns
        -------
        seen_task_samples: BaseDataSource
        unseen_task_samples: BaseDataSource
        '''

        if self.meta_model_stage.lower() == "train":
            return self.train(**kwargs)
        else:
            return self.inference(samples, **kwargs)

    def inference(self, samples, **kwargs):
        seen_task_samples = BaseDataSource(data_type=samples.data_type)
        unseen_task_samples = BaseDataSource(data_type=samples.data_type)
        seen_task_samples.x, unseen_task_samples.x = [], []

        if not self.backup_model:
            allocations = [self.seen_extractor.get(
                self.origin) for _ in samples.x]
            samples, models = self._task_remodeling(
                samples=samples, mappings=allocations)
        else:
            models = [Model(index=0, result={},
                            entry=self.backup_model, model=self.backup_model)]
            samples.inx = range(samples.num_examples())
            samples = [samples]

        tasks = []
        OOD_scores = []
        for inx, df in enumerate(samples):
            m = models[inx]
            if not isinstance(m, Model):
                continue
            if isinstance(m.model, str):
                evaluator = set_backend(estimator=self.base_model)
                if self.backup_model is None:
                    evaluator.load(m.model)
            else:
                evaluator = m.model
            InD_list, OoD_list, pred, ood_scores = self.ood_predict(
                evaluator, df.x, **kwargs)
            seen_task_samples.x.extend(InD_list)
            unseen_task_samples.x.extend(OoD_list)
            OOD_scores.extend(ood_scores)
            task = Task(entry=m.entry, samples=df)
            task.result = pred
            task.model = m
            tasks.append(task)
        res = self._inference_integrate(tasks)
        return (seen_task_samples, res, tasks), \
            (unseen_task_samples, OOD_scores)

    def ood_predict(self, evaluator, samples, **kwargs):
        data = self.preprocess_func(samples)
        evaluator.estimator.validator.test_loader = DataLoader(
            data,
            batch_size=1,
            shuffle=False,
            pin_memory=True)
        seg_model = evaluator.estimator.validator.model
        data_loader = evaluator.estimator.validator.test_loader

        OoD_list, InD_list = [], []
        predictions = []
        ood_scores = []

        seg_model.eval()
        evaluator.estimator.validator.evaluator.reset()
        for i, (sample, _) in enumerate(data_loader):
            image = sample["image"]
            if self.cuda:
                image = image.cuda()
            with torch.no_grad():
                _, output = seg_model(image)
            if self.cuda:
                torch.cuda.synchronize()
            pred = torch.max(output, 1)[1]

            glcm_features = self.calculate_glcm_features(
                pred.squeeze(0).cpu().numpy().astype(np.uint8))
            glcm_score = glcm_features['homogeneity']
            if glcm_score < self.glcm_thresh:
                LOGGER.info(f'glcm homogeneity score:{glcm_score}')
                OoD_list.append(samples[i])
                continue

            maxLogit = torch.max(output, 1)[0].unsqueeze(1)
            maxLogit = self.batch_min_max(maxLogit)
            softmaxDistance = self.get_softmaxDistance(output).unsqueeze(1)
            cosDistanceLoss = self.get_cosDistanceLoss(output, pred)

            maxLogit, softmaxDistance = maxLogit.mean(1, keepdim=True), \
                softmaxDistance.mean(1, keepdim=True)

            origin_shape = maxLogit.shape
            maxLogit, softmaxDistance, cosDistanceLoss = maxLogit.flatten(), \
                softmaxDistance.flatten(), cosDistanceLoss.flatten()

            effec_shape = maxLogit.shape[0]
            temp_x = torch.cat([maxLogit.reshape(effec_shape, 1),
                                softmaxDistance.reshape(effec_shape, 1),
                                cosDistanceLoss.reshape(effec_shape, 1)],
                               dim=1)

            OOD_pred = self.ood_model.predict(temp_x.cpu().numpy())
            OOD_pred_show = OOD_pred + 1
            OOD_pred_show = OOD_pred_show.reshape(origin_shape)

            for j in range(origin_shape[0]):
                OOD_score = (OOD_pred_show[j] == 1).sum(
                ) / (OOD_pred_show[j] != 0).sum()
                LOGGER.info(f'OOD_score:{OOD_score}')
                if OOD_score > self.OOD_thresh:
                    OoD_list.append(samples[i])
                    ood_scores.append(OOD_score)
                else:
                    InD_list.append(samples[i])
                    predictions.append(pred.data.cpu().numpy())

        return InD_list, OoD_list, predictions, ood_scores

    def train(self, **kwargs):
        ood_data_path = os.path.join(BaseConfig.data_path_prefix, 'ood_data')
        LOGGER.info("Start to generate ood data.")
        for task_group in self.seen_task_groups:
            self.generate_train_data(
                task_group.samples,
                ood_data_path,
                task_group.model.model,
                **kwargs)

        train_x, train_y = self.prepare_train_data(ood_data_path)
        lr_model = LogisticRegression()
        lr_model.fit(train_x, train_y)
        return FileOps.dump(lr_model, "lr_model.model")

    def generate_train_data(self, samples, ood_data_path, model_path,
                            **kwargs):
        self.base_model.load(model_path)
        data = self.preprocess_func(samples)
        self.base_model.validator.test_loader = DataLoader(
            data,
            batch_size=1,
            shuffle=False,
            pin_memory=True)
        seg_model = self.base_model.validator.model
        data_loader = self.base_model.validator.test_loader

        seg_model.eval()
        self.base_model.validator.evaluator.reset()

        OOD_gt_path = os.path.join(ood_data_path, 'OOD_gt')
        os.makedirs(OOD_gt_path, exist_ok=True)
        maxLogit_path = os.path.join(ood_data_path, 'maxLogit')
        os.makedirs(maxLogit_path, exist_ok=True)
        softmaxDistance_path = os.path.join(ood_data_path, 'softmaxDistance')
        os.makedirs(softmaxDistance_path, exist_ok=True)
        cosDistanceLoss_path = os.path.join(ood_data_path, 'cosDistanceLoss')
        os.makedirs(cosDistanceLoss_path, exist_ok=True)

        tbar = tqdm(data_loader, desc='\r')
        for _, (sample, image_name) in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.cuda:
                image = image.cuda()
                target = target.cuda()

            with torch.no_grad():
                _, output = seg_model(image)
            if self.cuda:
                torch.cuda.synchronize()

            seg_pred = torch.max(output, 1)[1]
            # pred colorize
            maxlogit = torch.max(output, 1)[0].unsqueeze(1)
            maxlogit = self.batch_min_max(maxlogit)
            softmaxDistance = self.get_softmaxDistance(output)
            cosloss = self.get_cosDistanceLoss(output, seg_pred)

            seg_pred[target > self.base_model.validator.num_class - 1] = 255
            for j in range(data_loader.batch_size):
                # store OOD_gt
                target_piece = self.sample_transform(
                    target[j].unsqueeze(0).unsqueeze(0), output.shape[2:])
                seg_pred_piece = self.sample_transform(
                    seg_pred[j].float().unsqueeze(0).unsqueeze(0),
                    output.shape[2:])

                OOD_gt = torch.zeros(target_piece.shape)
                if self.cuda:
                    OOD_gt = OOD_gt.cuda()
                OOD_gt[seg_pred_piece == target_piece] = 1
                OOD_gt[target_piece == 255] = 255
                OOD_gt_img_path = os.path.join(
                    OOD_gt_path, os.path.basename(image_name[j]))
                torchvision.utils.save_image(
                    OOD_gt, OOD_gt_img_path, normalize=True)

                # store maxLogit
                maxLogit_img_path = os.path.join(
                    maxLogit_path, os.path.basename(image_name[j]))
                torchvision.utils.save_image(
                    maxlogit[j], maxLogit_img_path, normalize=True)

                # store softmaxDistance
                softmaxDistance_img_path = os.path.join(
                    softmaxDistance_path, os.path.basename(image_name[j]))
                torchvision.utils.save_image(
                    softmaxDistance[j], softmaxDistance_img_path,
                    normalize=True)

                # store CosDistanceLoss
                cosDistanceLoss_img_path = os.path.join(
                    cosDistanceLoss_path, os.path.basename(image_name[j]))
                torchvision.utils.save_image(
                    cosloss[j], cosDistanceLoss_img_path, normalize=True)

    def prepare_train_data(self, data_path):
        LOGGER.info("Start to prepare ood train data.")

        transformed_images = []
        ood_gt_path = os.path.join(data_path, "OOD_gt")
        ood_gt_data = list(map(lambda x: os.path.join(ood_gt_path, x),
                               os.listdir(ood_gt_path)))
        maxLogit_path = os.path.join(data_path, "maxLogit")
        maxLogit_data = list(map(lambda x: os.path.join(maxLogit_path, x),
                                 os.listdir(maxLogit_path)))
        softmaxDistance_path = os.path.join(data_path, "softmaxDistance")
        softmaxDistance_data = list(map(lambda x:
                                        os.path.join(softmaxDistance_path, x),
                                        os.listdir(softmaxDistance_path)))
        cosDistanceLoss_path = os.path.join(data_path, "cosDistanceLoss")
        cosDistanceLoss_data = list(map(lambda x:
                                        os.path.join(cosDistanceLoss_path, x),
                                        os.listdir(cosDistanceLoss_path)))
        ood_gt_data.sort()
        maxLogit_data.sort()
        softmaxDistance_data.sort()
        cosDistanceLoss_data.sort()

        for _, (ood_gt, maxLogit, softmaxDistance, cosDistanceLoss) in \
            enumerate(zip(ood_gt_data, maxLogit_data,
                          softmaxDistance_data, cosDistanceLoss_data)):
            _maxlogit = Image.open(maxLogit).convert('RGB')
            _softmaxdistance = Image.open(softmaxDistance).convert('RGB')
            _cosdistanceloss = Image.open(cosDistanceLoss).convert('RGB')
            _target = Image.open(ood_gt)
            sample = {'maxLogit': _maxlogit,
                      'softmaxDistance': _softmaxdistance,
                      'cosDistanceLoss': _cosdistanceloss,
                      'label': _target}
            composed_transforms = transforms.Compose([ToTensor()])
            transformed_images.append((composed_transforms(sample), maxLogit))

        data_loader = DataLoader(
            transformed_images, batch_size=4, shuffle=True)

        train_x = torch.zeros([1, 3])
        train_y = torch.zeros(1)
        if self.cuda:
            train_x = train_x.cuda()
            train_y = train_y.cuda()

        tbar = tqdm(data_loader, desc='\r')
        for _, (sample, image_name) in enumerate(tbar):
            maxLogit = sample['maxLogit']
            softmaxDistance = sample['softmaxDistance']
            cosDistanceLoss = sample['cosDistanceLoss']
            OOD_gt = sample['label']
            if self.cuda:
                maxLogit = maxLogit.cuda()
                softmaxDistance = softmaxDistance.cuda()
                cosDistanceLoss = cosDistanceLoss.cuda()
                OOD_gt = OOD_gt.cuda()

            maxLogit = maxLogit.mean(1, keepdim=True)
            softmaxDistance = softmaxDistance.mean(1, keepdim=True)
            cosDistanceLoss = cosDistanceLoss.mean(1, keepdim=True)
            OOD_gt = OOD_gt.mean(1, keepdim=True)

            OOD_gt_show = OOD_gt + 1
            OOD_gt_show[OOD_gt_show == 256] = 0
            for j in range(OOD_gt.shape[0]):
                OOD_gt_show_name = image_name[j].replace(
                    'maxLogit', 'OOD_gt_show')
                os.makedirs(os.path.dirname(OOD_gt_show_name), exist_ok=True)
                torchvision.utils.save_image(
                    OOD_gt_show[j], OOD_gt_show_name, normalize=True, value_range=(0, 2))

            maxLogit = maxLogit.flatten()
            softmaxDistance = softmaxDistance.flatten()
            cosDistanceLoss = cosDistanceLoss.flatten()
            OOD_gt = OOD_gt.flatten()
            maxLogit = maxLogit[OOD_gt != 255]
            softmaxDistance = softmaxDistance[OOD_gt != 255]
            cosDistanceLoss = cosDistanceLoss[OOD_gt != 255]
            temp_y = OOD_gt[OOD_gt != 255]

            effec_shape = maxLogit.shape[0]
            temp_x = torch.cat([maxLogit.reshape(effec_shape, 1),
                                softmaxDistance.reshape(effec_shape, 1),
                                cosDistanceLoss.reshape(effec_shape, 1)],
                               dim=1)
            train_x = torch.cat([train_x, temp_x], dim=0)
            train_y = torch.cat([train_y, temp_y], dim=0)

        train_x = train_x[1:, :].cpu().numpy()
        train_y = train_y[1:].cpu().numpy()

        rus = RandomUnderSampler(random_state=0)
        return rus.fit_resample(train_x, train_y)

    def batch_min_max(self, img):
        max_value = torch.amax(img, [1, 2, 3]).unsqueeze(dim=1)
        min_value = torch.amin(img, [1, 2, 3]).unsqueeze(dim=1)

        [b, n, h, w] = img.shape
        img1 = img.reshape(b, -1)
        img2 = (img1 - min_value) / (max_value - min_value)
        img2 = img2.reshape([b, n, h, w])
        return img2

    def get_softmaxDistance(self, logits):
        seg_softmax_out = torch.nn.Softmax(dim=1)(logits.detach())
        distance, _ = torch.topk(seg_softmax_out, 2, dim=1)
        max_softmaxLogit = distance[:, 0, :, :]
        max2nd_softmaxLogit = distance[:, 1, :, :]
        return max_softmaxLogit - max2nd_softmaxLogit

    def get_cosDistanceLoss(self, logits, seg_pred):
        '''Calculate cos distance loss'''

        x = logits
        y = seg_pred.squeeze(0)
        b1, num, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(-1, num)
        y = y.reshape(-1)
        b = x.shape[0]
        int_batch = y.long()

        one_hot_batch = -torch.zeros(b, num)
        if self.cuda:
            one_hot_batch = one_hot_batch.cuda()
        one_hot_batch.scatter_(1, int_batch.unsqueeze(1), 1)
        norms = torch.norm(x, p=2, dim=1, keepdim=True) + 1e-7
        logit_norm = torch.div(x, norms)
        cos = torch.sum(torch.mul(logit_norm, one_hot_batch), dim=1)
        return cos.reshape(b1, h, w)

    def calculate_glcm_features(self, matrix,
                                distances=[1, 2, 3, 4, 5, 6],
                                angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                                properties=['contrast', 'homogeneity',
                                            'energy', 'correlation']):
        glcm = graycomatrix(matrix, distances, angles,
                            symmetric=True, normed=True)
        glcm_features = {prop: np.mean(graycoprops(glcm, prop))
                         for prop in properties}
        return glcm_features

    def _task_remodeling(self, samples: BaseDataSource, mappings: List):
        """
        Grouping based on assigned tasks
        """
        mappings = np.array(mappings)
        data, models = [], []
        d_type = samples.data_type
        for m in np.unique(mappings):
            task_df = BaseDataSource(data_type=d_type)
            _inx = np.where(mappings == m)
            if isinstance(samples.x, pd.DataFrame):
                task_df.x = samples.x.iloc[_inx]
            else:
                task_df.x = np.array(samples.x)[_inx]
            if d_type != "test":
                if isinstance(samples.x, pd.DataFrame):
                    task_df.y = samples.y.iloc[_inx]
                else:
                    task_df.y = np.array(samples.y)[_inx]
            task_df.inx = _inx[0].tolist()
            if samples.meta_attr is not None:
                task_df.meta_attr = np.array(samples.meta_attr)[_inx]
            data.append(task_df)
            # TODO: if m is out of index
            try:
                model = self.seen_models[m]
            except Exception as err:
                print(f"self.models[{m}] not exists. {err}")
                model = self.seen_models[0]
            models.append(model)
        return data, models

    def _inference_integrate(self, tasks):
        res = {}
        for task in tasks:
            res.update(dict(zip(task.samples.inx, task.result)))
        return np.array([z[1]
                        for z in sorted(res.items(), key=lambda x: x[0])])


class ToTensor(object):
    """Convert Image object in sample to Tensors."""

    def __call__(self, sample):
        maxLogit = sample['maxLogit']
        softmaxDistance = sample['softmaxDistance']
        cosDistanceLoss = sample['cosDistanceLoss']
        mask = sample['label']

        maxLogit = np.array(maxLogit).astype(np.float32).transpose((2, 0, 1))
        softmaxDistance = np.array(softmaxDistance).astype(
            np.float32).transpose((2, 0, 1))
        cosDistanceLoss = np.array(cosDistanceLoss).astype(
            np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32).transpose((2, 0, 1))

        maxLogit = torch.from_numpy(maxLogit).float()/255.0
        softmaxDistance = torch.from_numpy(softmaxDistance).float()/255.0
        cosDistanceLoss = torch.from_numpy(cosDistanceLoss).float()/255.0

        mask = torch.from_numpy(mask).float()

        return {'maxLogit': maxLogit,
                'softmaxDistance': softmaxDistance,
                'cosDistanceLoss': cosDistanceLoss,
                'label': mask}
