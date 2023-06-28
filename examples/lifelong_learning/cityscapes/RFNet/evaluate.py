import os
# os.environ["OUTPUT_URL"] = "s3://kubeedge/sedna-robo/kb_2"
# os.environ["MODEL_URLS"] = "s3://kubeedge/sedna-robo/kb/index.pkl"
# os.environ["TEST_DATASET_URL"] = "/home/lsq/RFNet/data_index/test.txt"
# os.environ["ORIGINAL_DATASET_URL"] = "/home/lsq/RFNet/data_index/test.txt"
# os.environ["KB_SERVER"] = "http://127.0.0.1:9020"

# os.environ["model_threshold"] = "0"
# os.environ["operator"] = "<"

from sedna.core.lifelong_learning import LifelongLearning
from sedna.datasources import TxtDataParse
from sedna.common.config import Context

from accuracy import accuracy
from interface import Estimator


def _load_txt_dataset(dataset_url):
    # use original dataset url
    original_dataset_url = Context.get_parameters('original_dataset_url', "")
    dataset_urls = dataset_url.split()
    dataset_urls = [
        os.path.join(
            os.path.dirname(original_dataset_url),
            dataset_url) for dataset_url in dataset_urls]
    return dataset_urls[:-1], dataset_urls[-1]


def eval():
    estimator = Estimator(num_class=31)
    eval_dataset_url = Context.get_parameters("test_dataset_url")
    eval_data = TxtDataParse(data_type="eval", func=_load_txt_dataset)
    eval_data.parse(eval_dataset_url, use_raw=False)

    task_allocation = {
        "method": "TaskAllocationSimple"
    }

    ll_job = LifelongLearning(estimator,
                              task_definition=None,
                              task_relationship_discovery=None,
                              task_allocation=task_allocation,
                              task_remodeling=None,
                              inference_integrate=None,
                              task_update_decision=None,
                              unseen_task_allocation=None,
                              unseen_sample_recognition=None,
                              unseen_sample_re_recognition=None
                              )

    ll_job.evaluate(eval_data, metrics=accuracy)


if __name__ == '__main__':
    print(eval())
