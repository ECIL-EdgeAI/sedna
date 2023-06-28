import os
# os.environ["OUTPUT_URL"] = "s3://kubeedge/sedna-robo/kb"
# os.environ["TRAIN_DATASET_URL"] = "/home/data/robo_dataset/data.txt"
# os.environ["ORIGINAL_DATASET_URL"] = "/home/data/robo_dataset/"
# os.environ["KB_SERVER"] = "http://127.0.0.1:9020"

# os.environ["CLOUD_KB_INDEX"] = "s3://kubeedge/sedna-robo/kb/index.pkl"
# os.environ["HAS_COMPLETED_INITIAL_TRAINING"] = "true"

from PIL import Image
from torchvision import transforms
from sedna.core.lifelong_learning import LifelongLearning
from sedna.common.config import Context, BaseConfig
from sedna.datasources import TxtDataParse

from interface import Estimator
from dataloaders import custom_transforms as tr


def preprocess(samples):
    data = zip(samples.x, samples.y)

    transformed_images = []
    for (x_data, y_data) in data:
        if len(x_data) == 2:
            img_path, depth_path = x_data
            _img = Image.open(img_path).convert(
                'RGB').resize((2048, 1024), Image.BILINEAR)
            _depth = Image.open(depth_path).resize(
                (2048, 1024), Image.BILINEAR)
        else:
            img_path = x_data[0]
            _img = Image.open(img_path).convert(
                'RGB').resize((2048, 1024), Image.BILINEAR)
            _depth = _img

        if y_data is None:
            y_data = _img
        else:
            y_data = Image.open(y_data).resize((2048, 1024), Image.BILINEAR)

        sample = {'image': _img, 'depth': _depth, 'label': y_data}
        composed_transforms = transforms.Compose([
            tr.CropBlackArea(),
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=1024, crop_size=768, fill=255),
            # tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        transformed_images.append((composed_transforms(sample), img_path))

    return transformed_images


def _load_txt_dataset(dataset_url):
    # use original dataset url
    original_dataset_url = Context.get_parameters('original_dataset_url', "")
    dataset_urls = dataset_url.split()
    dataset_urls = [
        os.path.join(
            os.path.dirname(original_dataset_url),
            dataset_url) for dataset_url in dataset_urls]
    return dataset_urls[:-1], dataset_urls[-1]


def train(estimator, train_data):
    task_definition = {
        "method": "TaskDefinitionSimple"
    }

    task_allocation = {
        "method": "TaskAllocationSimple"
    }

    unseen_sample_recognition = {
        "method": "OodIdentification",
        "param": {
            "preprocess_func": preprocess,
            "base_model": estimator,
            "stage": "train"
        }
    }

    ll_job = LifelongLearning(estimator,
                              task_definition=task_definition,
                              task_relationship_discovery=None,
                              task_allocation=task_allocation,
                              task_remodeling=None,
                              inference_integrate=None,
                              task_update_decision=None,
                              unseen_task_allocation=None,
                              unseen_sample_recognition=unseen_sample_recognition,
                              unseen_sample_re_recognition=None
                              )

    ll_job.train(train_data)


def run():
    estimator = Estimator()
    train_dataset_url = BaseConfig.train_dataset_url
    train_data = TxtDataParse(data_type="train", func=_load_txt_dataset)
    train_data.parse(train_dataset_url, use_raw=False)

    train(estimator, train_data)


if __name__ == '__main__':
    run()
