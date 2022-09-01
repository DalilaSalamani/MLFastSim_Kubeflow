# preprocess function loads the data and returns the array of the shower energies and the condition arrays
from typing import NamedTuple


def preprocess_new(nCells_z: int, nCells_r: int, nCells_phi: int, original_dim: int, min_energy: int, max_energy: int,
                   min_angle: int, max_angle: int, init_dir: str, checkpoint_dir: str, conv_dir: str, valid_dir: str,
                   gen_dir: str) -> NamedTuple('Variable_Details',
                                               [('energies_train_location', str), ('condE_train_location', str),
                                                ('condAngle_train_location', str), ('condGeo_train_location', str)]):
    import h5py
    import numpy as np
    energies_Train = []
    condE_Train = []
    condAngle_Train = []
    condGeo_Train = []
    # This example is trained using 2 detector geometries
    for geo in ['SciPb']:
        dirGeo = init_dir + geo + '/'
        # loop over the angles in a step of 10
        for angleParticle in range(min_angle, max_angle + 10, 10):
            fName = '%s_angle_%s.h5' % (geo, angleParticle)
            fName = dirGeo + fName
            #             read the HDF5 file
            h5 = h5py.File(fName, 'r')
            # loop over energies from min_energy to max_energy
            energyParticle = min_energy
            while (energyParticle <= max_energy):
                # scale the energy of each cell to the energy of the primary particle (in MeV units) 
                events = np.array(h5['%s' % energyParticle]) / (energyParticle * 1000)
                print(events.shape)
                #                 print(events.shape)

                energies_Train.append(events.reshape(len(events), original_dim))
                print("Done")
                # build the energy and angle condition vectors
                condE_Train.append([energyParticle / max_energy] * len(events))
                print("Done")
                condAngle_Train.append([angleParticle / max_angle] * len(events))
                print("Done")
                # build the geometry condition vector (1 hot encoding vector)
                if (geo == 'SiW'):
                    condGeo_Train.append([[0, 1]] * len(events))
                if (geo == 'SciPb'):
                    condGeo_Train.append([[1, 0]] * len(events))
                    print("Done")
                energyParticle *= 2
    print('possible Error')
    # return numpy arrays 
    energies_Train = np.concatenate(energies_Train)
    condE_Train = np.concatenate(condE_Train)
    condAngle_Train = np.concatenate(condAngle_Train)
    condGeo_Train = np.concatenate(condGeo_Train)
    energies_train_location = '/eos/user/g/gkohli/input_save/energies_train4.npy'
    np.save(energies_train_location, energies_Train)
    condE_train_location = '/eos/user/g/gkohli/input_save/condE_train.npy'
    np.save(condE_train_location, condE_Train)
    condAngle_train_location = '/eos/user/g/gkohli/input_save/condAngle_train.npy'
    np.save(condAngle_train_location, condAngle_Train)
    condGeo_train_location = '/eos/user/g/gkohli/input_save/condGeo_train.npy'
    np.save(condGeo_train_location, condGeo_Train)
    return energies_train_location, condE_train_location, condAngle_train_location, condGeo_train_location