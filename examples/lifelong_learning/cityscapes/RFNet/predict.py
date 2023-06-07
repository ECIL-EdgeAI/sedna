import time
import os
# os.environ["MODEL_URLS"] = "s3://kubeedge/sedna-robo/kb/index.pkl"
# os.environ["KB_SERVER"] = "http://127.0.0.1:9020"
# os.environ["test_data"] = "/data/test_data"
# os.environ["unseen_save_url"] = "s3://kubeedge/sedna-robo/unseen_samples/"
# os.environ["metadata_url"] = "s3://kubeedge/sedna-robo/metadata/"
# os.environ["OUTPUT_URL"] = "s3://kubeedge/sedna-robo/"

# os.environ["OOD_thresh"] = "0.01"
# os.environ["ramp_detection"] = "/home/lsq/RFNet/models/garden_2_GPU.pth"

from PIL import Image
from sedna.datasources import BaseDataSource
from sedna.core.lifelong_learning import LifelongLearning
from sedna.common.config import Context
from sedna.common.log import LOGGER

from interface import Estimator, preprocess_frames


def preprocess(samples):
    data = BaseDataSource(data_type="test")
    data.x = [samples]
    return data


def postprocess(samples):
    image_names, imgs = [], []
    for sample in samples:
        img = sample.get("image")
        image_names.append("{}.png".format(str(time.time())))
        imgs.append(img)

    return image_names, imgs


def init_ll_job():
    robo_skill = Context.get_parameters("robo_skill", "ramp_detection")
    estimator = Estimator(num_class=31,
                          save_predicted_image=True,
                          weight_path=Context.get_parameters(robo_skill),
                          merge=True)

    task_allocation = {
        "method": "TaskAllocationDefault"
    }
    unseen_task_allocation = {
        "method": "UnseenTaskAllocationDefault"
    }
    unseen_sample_recognition = {
        "method": "OodIdentification",
        "param": {
            "OOD_thresh": float(Context.get_parameters("OOD_thresh")),
            "OOD_model": Context.get_parameters("OOD_model"),
            "OOD_backup_model": Context.get_parameters(robo_skill),
            "preprocess_func": preprocess_frames,
            "base_model": Estimator,
            "stage": "inference"
        }
    }

    ll_job = LifelongLearning(
        estimator,
        unseen_estimator=unseen_task_processing,
        task_definition=None,
        task_relationship_discovery=None,
        task_allocation=task_allocation,
        task_remodeling=None,
        inference_integrate=None,
        task_update_decision=None,
        unseen_task_allocation=unseen_task_allocation,
        unseen_sample_recognition=unseen_sample_recognition,
        unseen_sample_re_recognition=None)
    return ll_job


def unseen_task_processing():
    return "Warning: unseen sample detected."


def predict():
    ll_job = init_ll_job()
    test_data_dir = Context.get_parameters("test_data")
    test_data = os.listdir(test_data_dir)
    test_data_num = len(test_data)
    count = 0

    # simulate a permenant inference service
    LOGGER.info(f"Inference service starts.")
    while True:
        for i, data in enumerate(test_data):
            LOGGER.info(f"Start to inference image {i + count + 1}")

            test_data_url = os.path.join(test_data_dir, data)
            img_rgb = Image.open(test_data_url).convert("RGB")
            sample = {'image': img_rgb, "depth": img_rgb, "label": img_rgb}
            predict_data = preprocess(sample)
            prediction, is_unseen, _ = ll_job.inference(predict_data)
            LOGGER.info(f"Image {i + count + 1} is unseen task: {is_unseen}")
            LOGGER.info(
                f"Image {i + count + 1} prediction result: {prediction}")
            time.sleep(1.0)

        count += test_data_num

if __name__ == '__main__':
    predict()
