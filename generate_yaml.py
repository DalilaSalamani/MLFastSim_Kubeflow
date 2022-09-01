import kfp
from kfp.components import func_to_container_op, InputPath, OutputPath, create_component_from_func
from kfp import dsl
from kubernetes import client as k8s_client
import pickle
from typing import NamedTuple

from pipeline_components import model_parameters
from pipeline_components.generate import generate
from pipeline_components.input_parameters import input_parameters
from pipeline_components.model_setup import model_setup
from pipeline_components.preprocess import preprocess_new
from pipeline_components.validate import validate


def generate_yaml():
    kfp.components.func_to_container_op(input_parameters, base_image='python:3.7',
                                        output_component_file='components_yaml'
                                                              '/input.yaml')
    kfp.components.func_to_container_op(model_parameters, base_image='gitlab-registry.cern.ch/ai-ml/kubeflow_images'
                                                                     '/tensorflow-notebook-gpu-2.1.0:v0.6.1-33',
                                        output_component_file='components_yaml/model_para.yaml')
    kfp.components.func_to_container_op(model_setup, base_image='gitlab-registry.cern.ch/gkohli/mlfastsim-kubeflow'
                                                                '-pipeline/kube_gkohli:cern_pipelinev2',
                                        output_component_file='components_yaml/model_setup.yaml')
    kfp.components.func_to_container_op(preprocess_new, base_image='gitlab-registry.cern.ch/gkohli/mlfastsim-kubeflow'
                                                                   '-pipeline/kube_gkohli:cern_pipelinev2',
                                        output_component_file='components_yaml/preprocess.yaml')
    kfp.components.func_to_container_op(generate,
                                        base_image='gitlab-registry.cern.ch/gkohli/mlfastsim-kubeflow-pipeline'
                                                   '/kube_gkohli:cern_pipelinev2',
                                        output_component_file='components_yaml/generate.yaml')
    kfp.components.func_to_container_op(validate,
                                        base_image='gitlab-registry.cern.ch/gkohli/mlfastsim-kubeflow-pipeline'
                                                   '/kube_gkohli:cern_pipelinev2',
                                        output_component_file='components_yaml/validate.yaml')
