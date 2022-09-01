from typing import NamedTuple


def model_input_parameters(original_dim: int, checkpoint_dir: str) -> NamedTuple('Variable_Details',
                                                                                 [('batch_size', int),
                                                                                  ('original_dim', int),
                                                                                  ('intermediate_dim1', int),
                                                                                  ('intermediate_dim2', int),
                                                                                  ('intermediate_dim3', int),
                                                                                  ('intermediate_dim4', int),
                                                                                  ('latent_dim', int),
                                                                                  ('epsilon_std', int), ('mu', int),
                                                                                  ('epochs', int), ('lr', float),
                                                                                  ('outActiv', str),
                                                                                  ('validation_split', float),
                                                                                  ('wReco', int), ('wkl', float),
                                                                                  ('ki', str), ('bi', str),
                                                                                  ('earlyStop', bool),
                                                                                  ('checkpoint_dir', str)]):
    """

    :param original_dim:
    :param checkpoint_dir:
    :return:
    """
    batch_size = 100
    original_dim = original_dim
    intermediate_dim1 = 100
    intermediate_dim2 = 50
    intermediate_dim3 = 20
    intermediate_dim4 = 14
    latent_dim = 10
    epsilon_std = 1
    mu = 0
    epochs = 100
    lr = 0.001
    outActiv = 'sigmoid'
    validation_split = 0.05
    wReco = original_dim
    wkl = 0.5
    ki = 'RandomNormal'
    bi = 'Zeros'
    earlyStop = True
    checkpoint_dir = checkpoint_dir

    return (
        batch_size, original_dim, intermediate_dim1, intermediate_dim2, intermediate_dim3, intermediate_dim4,
        latent_dim,
        epsilon_std, mu, epochs, lr, outActiv,
        validation_split, wReco, wkl, ki, bi, earlyStop, checkpoint_dir)
