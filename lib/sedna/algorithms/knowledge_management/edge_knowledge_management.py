import os
import time
import json
from datetime import datetime
import tempfile
import threading
import uuid
from queue import Queue

from watchdog.events import *
from sedna.common.log import LOGGER
from sedna.common.config import Context, BaseConfig
from sedna.common.class_factory import ClassType, ClassFactory
from sedna.common.file_ops import FileOps
from sedna.common.constant import KBResourceConstant

from .base_knowledge_management import BaseKnowledgeManagement

__all__ = ('EdgeKnowledgeManagement', )


@ClassFactory.register(ClassType.KM)
class EdgeKnowledgeManagement(BaseKnowledgeManagement):
    """
    Manage inference, knowledge base update, etc., at the edge.

    Parameters:
        ----------
    config: Dict
        parameters to initialize an object
    estimator: Instance
        An instance with the high-level API that greatly simplifies
        machine learning programming. Estimators encapsulate training,
        evaluation, prediction, and exporting for your model.
    """

    def __init__(self, config, seen_estimator, unseen_estimator):
        super(EdgeKnowledgeManagement, self).__init__(
            config, seen_estimator, unseen_estimator)

        self.edge_output_url = Context.get_parameters(
            "edge_output_url", KBResourceConstant.EDGE_KB_DIR.value)
        self.task_index = FileOps.join_path(
            self.edge_output_url, KBResourceConstant.KB_INDEX_NAME.value)
        self.local_unseen_save_url = FileOps.join_path(self.edge_output_url,
                                                       "unseen_samples")

        self.pinned_service_start = False
        self.unseen_sample_observer = None
        self.current_index_version = None
        self.lastest_index_version = None

        self.unseen_sample_queue = Queue(maxsize=100)
        self.seen_sample_queue = Queue(maxsize=100)
        self.unseen_sample_postprocess = None
        self.seen_sample_postprocess = None

    def update_kb(self, task_index):
        if isinstance(task_index, str):
            try:
                task_index = FileOps.load(task_index)
            except Exception as err:
                self.log.error(f"{err}")
                self.log.error(
                    "Load task index failed. KB deployment to the edge failed.")
                return None

        seen_task_index = task_index.get(self.seen_task_key)
        unseen_task_index = task_index.get(self.unseen_task_key)

        seen_extractor, seen_task_groups = self.save_task_index(
            seen_task_index, task_type=self.seen_task_key)
        unseen_extractor, unseen_task_groups = self.save_task_index(
            unseen_task_index, task_type=self.unseen_task_key)
        meta_estimators = self.save_meta_estimators(
            task_index["meta_estimators"])

        task_info = {
            self.seen_task_key: {
                self.task_group_key: seen_task_groups,
                self.extractor_key: seen_extractor
            },
            self.unseen_task_key: {
                self.task_group_key: unseen_task_groups,
                self.extractor_key: unseen_extractor
            },
            "meta_estimators": meta_estimators,
            "create_time": task_index.get("create_time", str(time.time()))
        }

        self.current_index_version = str(task_info.get("create_time"))
        self.lastest_index_version = self.current_index_version

        fd, name = tempfile.mkstemp()
        os.close(fd)
        FileOps.dump(task_info, name)
        return FileOps.upload(name, self.task_index)

    def save_task_index(self, task_index, task_type="seen_task"):
        extractor = task_index[self.extractor_key]
        if isinstance(extractor, str):
            extractor = FileOps.load(extractor)
        task_groups = task_index[self.task_group_key]

        model_upload_key = {}
        for task in task_groups:
            model_file = task.model.model
            save_model = FileOps.join_path(
                self.edge_output_url, task_type,
                os.path.basename(model_file)
            )
            if model_file not in model_upload_key:
                model_upload_key[model_file] = FileOps.download(
                    model_file, save_model)
            model_file = model_upload_key[model_file]

            task.model.model = save_model

            for _task in task.tasks:
                _task.model = FileOps.join_path(
                    self.edge_output_url, task_type, os.path.basename(model_file))
                sample_dir = FileOps.join_path(
                    self.edge_output_url, task_type,
                    f"{_task.samples.data_type}_{_task.entry}.sample")
                _task.samples.data_url = FileOps.download(
                    _task.samples.data_url, sample_dir)

        save_extractor = FileOps.join_path(
            self.edge_output_url, task_type,
            KBResourceConstant.TASK_EXTRACTOR_NAME.value
        )
        extractor = FileOps.dump(extractor, save_extractor)

        return extractor, task_groups

    def save_meta_estimators(self, meta_estimator_index):
        meta_estimators = {}
        for meta_estimator_name, meta_estimator in meta_estimator_index.items():
            meta_estimator_url = FileOps.join_path(
                self.edge_output_url, "meta_estimators",
                os.path.basename(meta_estimator))

            meta_estimator = FileOps.download(
                meta_estimator, meta_estimator_url)
            meta_estimators[meta_estimator_name] = meta_estimator
        return meta_estimators

    def save_unseen_samples(self, samples, **kwargs):
        ood_predictions = kwargs.get("unseen_params")[0]
        ood_scores = kwargs.get("unseen_params")[1]

        for i, sample in enumerate(samples.x):
            sample_id = str(uuid.uuid4())
            if isinstance(sample, dict):
                img = sample.get("image")
            else:
                img = sample[0]
            unseen_sample_info = (sample_id, img, ood_predictions[i],
                                  ood_scores[i])
            self.unseen_sample_queue.put(unseen_sample_info)

    def save_seen_samples(self, samples, results, **kwargs):
        for i, sample in enumerate(samples.x):
            sample_id = str(uuid.uuid4())
            if isinstance(sample, dict):
                img = sample.get("image")
            else:
                img = sample[0]

            seen_sample_info = (sample_id, img, results[i])
            self.seen_sample_queue.put(seen_sample_info)

    def start_services(self):
        UnseenSampleThread(self.unseen_sample_queue,
                           post_process=self.unseen_sample_postprocess).start()
        SeenSampleThread(self.seen_sample_queue,
                         post_process=self.seen_sample_postprocess).start()
        ModelHotUpdateThread(self).start()


class ModelHotUpdateThread(threading.Thread):
    """Hot task index loading with multithread support"""
    MODEL_MANIPULATION_SEM = threading.Semaphore(1)

    def __init__(self,
                 edge_knowledge_management,
                 callback=None
                 ):
        model_check_time = int(Context.get_parameters(
            "MODEL_POLL_PERIOD_SECONDS", "30")
        )
        if model_check_time < 1:
            LOGGER.warning("Catch an abnormal value in "
                           "`MODEL_POLL_PERIOD_SECONDS`, fallback with 30")
            model_check_time = 30
        self.edge_knowledge_management = edge_knowledge_management
        self.check_time = model_check_time
        self.callback = callback

        super(ModelHotUpdateThread, self).__init__()

        LOGGER.info(f"Model hot update service starts.")

    def run(self):
        while True:
            time.sleep(self.check_time)
            if not self.edge_knowledge_management.current_index_version:
                continue

            latest_task_index = Context.get_parameters("MODEL_URLS")
            if not latest_task_index:
                continue
            latest_task_index = FileOps.load(latest_task_index)
            self.edge_knowledge_management.lastest_index_version = str(
                latest_task_index.get("create_time"))


class UnseenSampleThread(threading.Thread):
    def __init__(self, unseen_sample_queue, **kwargs):
        self.check_time = 1
        self.unseen_sample_queue = unseen_sample_queue
        self.local_unseen_dir = os.path.join(BaseConfig.data_path_prefix,
                                             "unseen_samples")
        self.unseen_save_url = Context.get_parameters("unseen_save_url",
                                                      self.local_unseen_dir)
        local_metadata_dir = os.path.join(BaseConfig.data_path_prefix,
                                          "metadata")
        self.metadata_dir = Context.get_parameters("metadata_url",
                                                   local_metadata_dir)
        self.post_process = kwargs.get("post_process")

        os.makedirs(self.local_unseen_dir, exist_ok=True)
        if not FileOps.is_remote(self.unseen_save_url):
            os.makedirs(self.unseen_save_url, exist_ok=True)
        if not FileOps.is_remote(self.metadata_dir):
            os.makedirs(self.metadata_dir, exist_ok=True)

        self.unseen_sample_metadata = self.init_unseen_metadata_template()
        super(UnseenSampleThread, self).__init__()
        LOGGER.info(f"Unseen sample upload service starts.")

    def run(self):
        while True:
            time.sleep(self.check_time)
            sample_id, img, ood_pred, ood_score = self.unseen_sample_queue.get()
            unseen_sample_url = self.upload_unseen_sample(img, ood_pred)
            self.upload_meta_data(sample_id, ood_score, unseen_sample_url)
            LOGGER.info(f"Upload unseen sample to {unseen_sample_url}")
            self.unseen_sample_queue.task_done()

    def init_unseen_metadata_template(self):
        unseen_sample_metadata = {
            "sample_id": "",
            "sample_url": "",
            "sample_timestamp": "",
            "unseen_sample_info": {
                "unseen_task_recognition": "",
                "unseen_task_processing": "急停并等待人工介入"
            }
        }
        return unseen_sample_metadata

    def upload_unseen_sample(self, img, ood_pred):
        if not callable(self.post_process):
            LOGGER.info("Unseen sample post processing is not callable.")
            return

        if not isinstance(img, str):
            image_name = "{}.png".format(str(time.time()))
        else:
            image_name = os.path.basename(img)

        local_save_url, color_local_save_url = self.post_process(
            self.local_unseen_dir, img, ood_pred, image_name)

        FileOps.upload(color_local_save_url,
                       os.path.join(self.unseen_save_url,
                                    os.path.basename(color_local_save_url)))
        return FileOps.upload(local_save_url,
                              os.path.join(self.unseen_save_url, image_name))

    def upload_meta_data(self, sample_id, ood_score, unseen_sample_url,
                         metadata_suffix="json"):
        sample_name = os.path.split(unseen_sample_url)[-1]
        sample_name = os.path.splitext(sample_name)[0]
        sample_timestamp = str(datetime.now())

        self.unseen_sample_metadata["sample_id"] = sample_id
        self.unseen_sample_metadata["sample_url"] = unseen_sample_url
        self.unseen_sample_metadata["sample_timestamp"] = sample_timestamp
        self.unseen_sample_metadata["unseen_sample_info"]["unseen_task_recognition"] = ood_score

        fd, name = tempfile.mkstemp()
        os.close(fd)

        if metadata_suffix == "json":
            metadata_json_str = json.dumps(
                self.unseen_sample_metadata, indent=4, ensure_ascii=False)
            with open(name, 'w') as json_file:
                json_file.write(metadata_json_str)
        else:
            FileOps.dump(self.unseen_sample_metadata, name)

        metadata_url = FileOps.join_path(
            self.metadata_dir,
            "{}.metadata.{}".format(sample_name, metadata_suffix))
        FileOps.upload(name, metadata_url)


class SeenSampleThread(threading.Thread):
    def __init__(self, seen_sample_queue, **kwargs):
        self.check_time = 1
        self.seen_sample_queue = seen_sample_queue
        self.local_seen_dir = os.path.join(
            BaseConfig.data_path_prefix, "seen_samples")
        self.seen_save_url = Context.get_parameters(
            "seen_save_url", self.local_seen_dir)
        self.post_process = kwargs.get("post_process")

        os.makedirs(self.local_seen_dir, exist_ok=True)
        if not FileOps.is_remote(self.seen_save_url):
            os.makedirs(self.seen_save_url, exist_ok=True)

        super(SeenSampleThread, self).__init__()
        LOGGER.info(f"Seen sample saving service starts.")

    def run(self):
        while True:
            time.sleep(self.check_time)
            _, img, res = self.seen_sample_queue.get()
            self.upload_seen_sample(img, res)
            self.seen_sample_queue.task_done()

    def upload_seen_sample(self, img, res):
        if not callable(self.post_process):
            LOGGER.info("Seen sample post processing is not callable.")
            return

        if not isinstance(img, str):
            image_name = "{}.png".format(str(time.time()))
        else:
            image_name = os.path.basename(img)

        local_save_url = self.post_process(
            self.local_seen_dir, img, res, image_name)
        if not isinstance(local_save_url, str):
            for img_url in local_save_url:
                image_name = os.path.basename(img_url)
                FileOps.upload(img_url,
                               os.path.join(self.seen_save_url, image_name))
        else:
            image_name = os.path.basename(local_save_url)
            FileOps.upload(img_url,
                           os.path.join(self.seen_save_url, image_name))
