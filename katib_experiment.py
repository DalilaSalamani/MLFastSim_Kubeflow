import argparse
import kubernetes
import pathlib2
import yaml
import json
import kfp
import time


def edit_template(
        src, dst,
        experiment_name, namespace, run_id, ):
    with open(src, 'r') as f:
        template = f.read()

    template = template.replace('EXPERIMENT_NAME', experiment_name)
    template = template.replace('NAMESPACE', namespace)

    with open(dst, 'w') as f:
        f.write(template)


def create_experiment(client, yaml_filepath, namespace):
    print('Load template for submission')
    with open(yaml_filepath, 'r') as f:
        experiment_spec = yaml.load(f, Loader=yaml.FullLoader)
        print(json.dumps(experiment_spec, indent=2))

    client.create_namespaced_custom_object(
        group='kubeflow.org',
        version='v1beta1',
        namespace=namespace,
        plural='experiments',
        body=experiment_spec,
    )


def delete_experiment(name, namespace):
    print('Delete experiment:', name)
    k8s_co_client.delete_namespaced_custom_object(
        group='kubeflow.org',
        version='v1beta1',
        namespace=namespace,
        plural='experiments',
        name=name,
        body=kubernetes.client.V1DeleteOptions()
    )


def to_csv(data, columns):
    csv_table = ''
    for row in data:
        csv_row = []
        for name in columns:
            csv_row.append(row[name])
        csv_table += ','.join(csv_row) + '\n'
    return csv_table


def write_results_to_ui(results):
    metadata = {
        'outputs': [
            {
                'type': 'markdown',
                'storage': 'inline',
                'source': '# Optimal trial 🏆',
            }, {
                'type': 'table',
                'storage': 'inline',
                'format': 'csv',
                'header': ['name', 'value'],
                'source': to_csv(results['parameterAssignments'], ['name', 'value']),
            }, {
                'type': 'table',
                'storage': 'inline',
                'format': 'csv',
                'header': ['name', 'latest', 'max', 'min'],
                'source': to_csv(results['observation']['metrics'], ['name', 'latest', 'max', 'min']),
            }]
    }

    with open('/mlpipeline-ui-metadata.json', 'w') as f:
        json.dump(metadata, f)


def get_model_path(results, s3_bucket, run_id):
    model_path = f'{s3_bucket}/{run_id}/'
    values = [pa['value'] for pa in results['parameterAssignments']]
    model_path += '_'.join(values)
    model_path += '.pt'
    print('Model path:', model_path)
    return model_path


print('Parse arguments')
parser = argparse.ArgumentParser(description='Train Params')
parser.add_argument('--id', type=str)
parser.add_argument('--s3-bucket', type=str)
parser.add_argument('--data-train', type=str)
parser.add_argument('--data-val', type=str)
parser.add_argument('--data-test', type=str)
parser.add_argument('--data-config', type=str)
parser.add_argument('--network-config', type=str)
parser.add_argument('--delete-experiment', type=str)
parser.add_argument('--optimal-model-path', type=str)
args = parser.parse_args()
print('Args:', vars(args))

name = f'jec-katib-{args.id}'
namespace = kfp.Client().get_user_namespace()

edit_template(
    src='katib.yaml',
    dst='katib_changed.yaml',
    experiment_name=name,
    namespace=namespace,
)

print('Load incluster config')
kubernetes.config.load_incluster_config()

print('Obtain CO client')
k8s_co_client = kubernetes.client.CustomObjectsApi()

print('Create katib experiment')
create_experiment(k8s_co_client, 'katib_.yaml', namespace)

while True:
    time.sleep(5)

    resource = k8s_co_client.get_namespaced_custom_object(
        group='kubeflow.org',
        version='v1beta1',
        namespace=namespace,
        plural='experiments',
        name=name
    )

    status = resource['status']
    print(status)

    if 'completionTime' in status.keys():
        if status['completionTime']:
            optimal_trial = status['currentOptimalTrial']
            print('Optimal trial')
            print(json.dumps(optimal_trial, indent=2))
            break

write_results_to_ui(optimal_trial)

optimal_model_path = get_model_path(optimal_trial, args.s3_bucket, args.id)
pathlib2.Path(args.optimal_model_path).parent.mkdir(parents=True)
pathlib2.Path(args.optimal_model_path).write_text(optimal_model_path)

if args.delete_experiment == 'True':
    delete_experiment(name, namespace)
