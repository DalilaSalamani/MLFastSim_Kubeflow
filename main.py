import argparse
import http

import kfp
from kfp import dsl
from kubernetes import client as k8s_client

eos_host_path = k8s_client.V1HostPathVolumeSource(path='/var/eos')
eos_volume = k8s_client.V1Volume(name='eos', host_path=eos_host_path)
eos_volume_mount = k8s_client.V1VolumeMount(name=eos_volume.name, mount_path='/eos')

krb_secret = k8s_client.V1SecretVolumeSource(secret_name='krb-secret')
krb_secret_volume = k8s_client.V1Volume(name='krb-secret-vol', secret=krb_secret)
krb_secret_volume_mount = k8s_client.V1VolumeMount(name=krb_secret_volume.name, mount_path='/secret/krb-secret-vol')


def load_cookies(cookie_file, domain):
    cookiejar = http.cookiejar.MozillaCookieJar(cookie_file)
    cookiejar.load()
    for cookie in cookiejar:
        if cookie.domain == domain:
            cookies = f'{cookie.name}={cookie.value}'
            break
    return cookies


def get_pipeline(name):
    @dsl.pipeline(
        name=name,
        description='ML first).'
    )
    def ml_pipeline_first():
        data_dir = input_parameters_comp() \
            .add_volume(krb_secret_volume) \
            .add_volume_mount(krb_secret_volume_mount) \
            .add_volume(eos_volume) \
            .add_volume_mount(eos_volume_mount)

        preprocessed_input = preprocess_comp(data_dir.outputs['nCells_z'], data_dir.outputs['nCells_r'],
                                             data_dir.outputs['nCells_phi'], data_dir.outputs['original_dim'],
                                             data_dir.outputs['min_energy'], data_dir.outputs['max_energy'],
                                             data_dir.outputs['min_angle'], data_dir.outputs['max_angle'],
                                             data_dir.outputs['init_dir'], data_dir.outputs['checkpoint_dir'],
                                             data_dir.outputs['conv_dir'], data_dir.outputs['valid_dir'],
                                             data_dir.outputs['gen_dir']) \
            .add_volume(krb_secret_volume) \
            .add_volume_mount(krb_secret_volume_mount) \
            .add_volume(eos_volume) \
            .add_volume_mount(eos_volume_mount)
        model_instantations = model_input_parameters_comp(data_dir.outputs['original_dim'],
                                                          data_dir.outputs['checkpoint_dir']) \
            .add_volume(krb_secret_volume) \
            .add_volume_mount(krb_secret_volume_mount) \
            .add_volume(eos_volume) \
            .add_volume_mount(eos_volume_mount)
        generate = generate_comp(data_dir.outputs['max_energy'], model_instantations.outputs['checkpoint_dir'],
                                 data_dir.outputs['gen_dir'],
                                 model_instantations.outputs['batch_size'], model_instantations.outputs['original_dim'],
                                 model_instantations.outputs['latent_dim'], model_instantations.outputs['epsilon_std'],
                                 model_instantations.outputs['mu'], model_instantations.outputs['epochs'],
                                 model_instantations.outputs['lr'], model_instantations.outputs['outActiv'],
                                 model_instantations.outputs['validation_split'], model_instantations.outputs['wReco'],
                                 model_instantations.outputs['wkl'], model_instantations.outputs['ki'],
                                 model_instantations.outputs['bi'], model_instantations.outputs['earlyStop']) \
            .add_volume(krb_secret_volume) \
            .add_volume_mount(krb_secret_volume_mount) \
            .add_volume(eos_volume) \
            .add_volume_mount(eos_volume_mount)
        model_setup = model_setup_comp(model_instantations.outputs['batch_size'],
                                       model_instantations.outputs['original_dim'],
                                       model_instantations.outputs['intermediate_dim1'],
                                       model_instantations.outputs['intermediate_dim2'],
                                       model_instantations.outputs['intermediate_dim3'],
                                       model_instantations.outputs['intermediate_dim4'],
                                       model_instantations.outputs['latent_dim'],
                                       model_instantations.outputs['epsilon_std'],
                                       model_instantations.outputs['mu'], model_instantations.outputs['epochs'],
                                       model_instantations.outputs['lr'], model_instantations.outputs['outActiv'],
                                       model_instantations.outputs['validation_split'],
                                       model_instantations.outputs['wReco'],
                                       model_instantations.outputs['wkl'], model_instantations.outputs['ki'],
                                       model_instantations.outputs['bi'], model_instantations.outputs['earlyStop'],
                                       model_instantations.outputs['checkpoint_dir'],
                                       preprocessed_input.outputs['energies_train_location'],
                                       preprocessed_input.outputs['condE_train_location'],
                                       preprocessed_input.outputs['condAngle_train_location'],
                                       preprocessed_input.outputs['condGeo_train_location']) \
            .add_volume(krb_secret_volume) \
            .add_volume_mount(krb_secret_volume_mount) \
            .add_volume(eos_volume) \
            .add_volume_mount(eos_volume_mount)
        validate = validate_comp(generate.outputs['generate_data'], data_dir.outputs['nCells_z'],
                                 data_dir.outputs['nCells_r'], data_dir.outputs['nCells_phi'],
                                 data_dir.outputs['save_dir'], data_dir.outputs['max_energy'],
                                 model_instantations.outputs['checkpoint_dir'], data_dir.outputs['init_dir'],
                                 data_dir.outputs['gen_dir'], data_dir.outputs['save_dir'],
                                 model_instantations.outputs['original_dim'], data_dir.outputs['valid_dir']) \
            .add_volume(krb_secret_volume) \
            .add_volume_mount(krb_secret_volume_mount) \
            .add_volume(eos_volume) \
            .add_volume_mount(eos_volume_mount)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pipeline Params')
    parser.add_argument('--namespace', type=str, default='gkohli',
                        help='Kubeflow namespace to run pipeline in')
    parser.add_argument('--pipeline_name', type=str, default='test_run',
                        help='Kubeflow namespace to run pipeline in')
    parser.add_argument('--experiment_name', type=str, default='geant4-experiment',
                        help='name for KFP experiment on Kubeflow')
    args = parser.parse_args()

    # Define pipeline variables
    pipeline_file = args.pipeline_name + '.yaml'
    experiment_name = args.experiment_name

    # Import pipeline components
    input_parameters_comp = kfp.components.load_component_from_file('component_yaml/input.yaml')
    preprocess_comp = kfp.components.load_component_from_file('component_yaml/preprocess.yaml')
    model_input_parameters_comp = kfp.components.load_component_from_file('component_yaml/model_para.yaml')
    generate_comp = kfp.components.load_component_from_file('component_yaml/generate.yaml')
    model_setup_comp = kfp.components.load_component_from_file('component_yaml/model_setup.yaml')
    validate_comp = kfp.components.load_component_from_file('component_yaml/validate.yaml')

    # Get pipeline instance
    get_pipeline(args.pipeline_name)
    cookies = load_cookies(cookie_file='cookies.txt', domain='ml-staging.cern.ch')

    # Load Kubeflow pipeline client
    client = kfp.Client(host='https://ml-staging.cern.ch/pipeline', cookies=cookies)
    import yaml

    workflow = kfp.compiler.Compiler().compile(get_pipeline(args.pipeline_name), pipeline_file)


    def post_process(pipeline_file, outfile):
        with open(pipeline_file, "r") as stream:
            pip_dict = yaml.safe_load(stream)

        copy_command = 'cp /secret/krb-secret-vol/krb5cc_1000 /tmp/krb5cc_1000'
        chmod_command = 'chmod 600 /tmp/krb5cc_1000'

        for template in pip_dict['spec']['templates']:
            if 'container' in template.keys():
                component_command_list = template['container']['command'][2].split('\n')
                component_command_list.insert(2, copy_command)
                component_command_list.insert(3, chmod_command)

                # Check EOS access with this command
                # component_command_list.insert(4, 'ls -l /eos/user/d/dgolubov')
                joined_string = '\n'.join(component_command_list)

                template['container']['command'][2] = joined_string

        with open(outfile, 'w') as outfile:
            yaml.dump(pip_dict, outfile, default_flow_style=False)


    post_process(pipeline_file, pipeline_file)
    client.upload_pipeline(pipeline_file, args.pipeline_name)
    exp = client.create_experiment(name=args.experiment_name)
    run = client.run_pipeline(exp.id, args.pipeline_name, pipeline_file)

    print('Deployed', args.pipeline_name)
