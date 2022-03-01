import os 
from absl import logging 
from tfx.orchestration.kubeflow import kubeflow_dag_runner 
from pipeline import create_pipeline
import time 
suffix = int(time.time())

PIPELINE_NAME = f'salary_prediction_{suffix}'
PIPELINE_ROOT = 'gs://singular_willow_pipeline/metadata'
DATA_PATH = 'gs://singular_willow_pipeline/data'
SERVING_DIR = 'gs://singular_willow_pipeline/models'
#temp = 'gs://singular_willow_pipeline/temp'


'''
beam_pipeline_args = [
    "--runner=DataflowRunner",
    "--experiments=shuffle_mode=auto",
    "--project=singular-willow-339022",
    f"--temp_location={temp}",
    f"--region=us-central1",
    "--disk_size_gb=50"
]
'''

def run():
    
    metadata_config = kubeflow_dag_runner.get_default_kubeflow_metadata_config()
    tfx_image = 'gcr.io/singular-willow-339022/mlimage'
    runner_config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
        kubeflow_metadata_config=metadata_config, tfx_image=tfx_image
    )

    kubeflow_dag_runner.KubeflowDagRunner(config=runner_config).run(
        create_pipeline(
            pipeline_name=PIPELINE_NAME,
            pipeline_root=PIPELINE_ROOT,
            serving_dir=SERVING_DIR,
            data_path=DATA_PATH
        )
    )


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    run()

#tfx pipeline create --pipeline-path=kubeflow_dag_runner.py --endpoint=https://6c04783b0063a2f7-dot-us-central1.pipelines.googleusercontent.com/