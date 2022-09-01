from typing import NamedTuple


def input_parameters() -> NamedTuple('Variable_Details',[('nCells_z', int), ('nCells_r', int), ('nCells_phi', int), ('original_dim', int),
                                      ('min_energy', int), ('max_energy', int), ('min_angle', int), ('max_angle', int),
                                      ('init_dir', str), ('checkpoint_dir', str), ('conv_dir', str), ('valid_dir', str),
                                      ('gen_dir', str), ('save_dir', str)]):
    """
    Handling Input Parameters of the project
    :return: The global input variables
    """
    nCells_z, nCells_r = 45, 18
    nCells_phi = 50
    min_energy = 1
    max_energy = 1
    min_angle = 50
    max_angle = 50
    init_dir = '/eos/user/g/gkohli/'
    checkpoint_dir = '/eos/user/g/gkohli/checkpoint/'
    conv_dir = '/eos/user/g/gkohli/conversion/'
    valid_dir = '/eos/user/g/gkohli/validation/'
    gen_dir = '/eos/user/g/gkohli/generation/'
    save_dir = '/eos/user/g/gkohli/visualisations/'

    return (nCells_z, nCells_r, nCells_phi, nCells_z * nCells_r * nCells_phi, min_energy, max_energy,
            min_angle, max_angle, init_dir, checkpoint_dir, conv_dir, valid_dir, gen_dir, save_dir)
