"""
convert.py
==========

Functions for converting between different file formats used in the analysis pipeline.

Functions
---------

"""

import os
import uproot
import numpy as np
import awkward as ak
import qgpy.utils
from typing import Dict, Any, Union
import logging
import tensorflow as tf
import pickle

def awkward_to_tensor(array: ak.Array) -> Union[tf.RaggedTensor, tf.Tensor]:
    """Converts an awkward `ak.Array` to a Tensorflow `tf.RaggedTensor` or tf.Tensor. The output is a `tf.RaggedTensor` 
    if the array has a dimension greater than 1, otherwise it is a `tf.Tensor`. The number of dimensions of the array 
    gives the number of dimensions of the output.

    Args:
        array (ak.Array): awkward ak.Array to be converted. Can have a single or multiple dimensions.

    Returns:
        tf.RaggedTensor or tf.Tensor: `tf.RaggedTensor` if the array dimension is greater than 1, else `tf.Tensor`.
    """
    if array.ndim == 1:
        return tf.constant(array.to_list())
    elif array.ndim == 2:
        row_lengths = ak.num(array, axis=1).to_list()
        return tf.RaggedTensor.from_row_lengths(ak.flatten(array, axis=None).to_list(), row_lengths=row_lengths, validate=False)
    else:
        nested_row_lengths = [ak.flatten(ak.num(array, axis=ax), axis=None).to_list()
                              for ax in range(1, array.ndim)]
        return tf.RaggedTensor.from_nested_row_lengths(ak.flatten(
            array, axis=None).to_list(), nested_row_lengths=nested_row_lengths, validate=False)

def delphes_to_jidenn_root(delphes_file: str, jidenn_file: str) -> None:
    """
    Convert Delphes root format to JIDENN accepted root format.

    Parameters
    ----------
    delphes_file : str
        The path to the input Delphes root file.
    jidenn_file : str
        The path to the output JIDENN root file.

    Returns
    -------
    ret : None
    """

    # Create a logger instance.
    outdir = delphes_file.rsplit('/', 1)[0]
    print(f"Output directory: {outdir}")
    logger = qgpy.utils.create_logger('convert', outdir = outdir)
    logger.info("Starting the Delphes to JIDENN-input conversion...")

    njets = 2

    store = {}

    # Open the ROOT file.
    with uproot.open(delphes_file) as file:

        # Access the TTree named "Delphes"
        tree = file["Delphes"]
        # List all keys in the root file
        tree.show()
        keys = tree.keys()
        logger.info("Keys in the root file:")
        for key in keys:
            logger.info(f'{key}, {type(tree[key])}')

        store['jets_pt'] = tree["Jet/Jet.PT"].array()
        store['jets_eta'] = tree["Jet/Jet.Eta"].array()
        store['jets_phi'] = tree["Jet/Jet.Phi"].array()
        store['jets_PartonTruthLabelID'] = ak.values_astype(tree["Jet/Jet.Flavor"].array(), 'int32')

        # Calculate the jet energy from pt, eta, phi, and mass.
        jet_mass = tree["Jet/Jet.Mass"].array()
        total_momentum = store['jets_pt'] * np.cosh(store['jets_eta'])
        store['jets_energy'] = ak.Array(np.sqrt(total_momentum**2 + jet_mass**2))

        # Extract the jet_tau<x> values.
        jet_tau = tree["Jet/Jet.Tau[5]"].array()
        # From the last dimension of the jet_tau array, extract the first element.
        for i in [1, 2, 3, 4]:
            store[f'jets_tau{i}'] = jet_tau[..., i]

        # Extract the number of particles in a jet.
        store['jets_nparticles'] = tree["Jet/Jet.NCharged"].array() + tree["Jet/Jet.NNeutrals"].array()

        # Copy the array store['jets_pt'] and fill the copy with zeros.
        store['jets_sdmass'] = ak.zeros_like(store['jets_pt'])

    # Create a new root file and a new TTree to store the selected branches.
    with uproot.recreate(jidenn_file) as new_file:
        new_file["Delphes"] = store

    # Print the values in the jets_tau1 branch.
    with uproot.open(jidenn_file) as file:
        tree = file["Delphes"]
        logger.info("Values in the jets_tau1 branch:")
        logger.info(tree["jets_tau1"].array())

    # Return.
    return

def delphes_to_tf_dataset(job_dir: str, delphes_file: str, dataset_dir: str) -> None:
        
    # Define constituent properties to extract.
    constituent_properties = {
        "ParticleFlowCandidate/ParticleFlowCandidate.PT": "jets_c_pt",
        "ParticleFlowCandidate/ParticleFlowCandidate.Eta": "jets_c_eta",
        "ParticleFlowCandidate/ParticleFlowCandidate.Phi": "jets_c_phi",
        "ParticleFlowCandidate/ParticleFlowCandidate.E": "jets_c_e",
        "ParticleFlowCandidate/ParticleFlowCandidate.Charge": "jets_c_charge",
    }

    # Create a logger instance.
    outdir = delphes_file.rsplit('/', 1)[0]
    logger = qgpy.utils.create_logger('convert', outdir = outdir)
    logger.info("Starting the Delphes to JIDENN conversion...")

    # Create a dictionary to store the jet properties.
    store = {v: [] for v in constituent_properties.values()}

    # Open the ROOT file.
    with uproot.open(delphes_file) as file:

        # Access the TTree named "Delphes"
        tree = file["Delphes"]
        # List all keys in the root file
        keys = tree.keys()
        logger.info("Keys in the root file:")
        for key in keys:
            logger.debug(f'{key}, {type(tree[key])}')

        # Extract the branches from the tree
        jet_constituents = tree["Jet/Jet.Constituents"].array()
        constituent_funique_ids = tree["ParticleFlowCandidate/ParticleFlowCandidate.fUniqueID"].array()
        constituent_properties_arrays = {
            variable: tree[prop].array() for prop, variable in constituent_properties.items()
        }
        logger.info(jet_constituents)

        # Create an awkward array of jet constituent properties (pt, eta, phi, E, and charge),
        # belonging to each jet in the event.
        # The first dimension is the events, the second is the jets, and the third is the constituents.
        # Each jet has an array of references to the ParticleFlowCandidate objects; it is stored in the "refs" attribute within the Jet/Jet.Constituents branch.
        # The constituents' properties are stored in the ParticleFlowCandidate/ParticleFlowCandidate.PT/Eta/Phi/E/Charge branches.

        # Number of events in the root file.
        nevents = len(jet_constituents)
        logger.info(f"Number of events in the root file: {nevents}")

        # Loop over all events in the root file.
        for evt in range(nevents):
            # Print the progress every 10 events.
            if evt % 100 == 0:
                logger.info(f"Processing event {evt} of {nevents}")

            # Create a dictionary to store the jet properties for the current event.
            event_store = {v: [] for v in constituent_properties.values()}

            # Map the unique IDs of the constituents to their indices in the constituent_funique_ids array.
            id_to_index = {id_value: index for index, id_value in enumerate(constituent_funique_ids[evt])}

            njets = len(jet_constituents[evt])
            logger.debug(f"Number of jets in event {evt}: {njets}")

            # Loop over jets in the event.
            for i_jet in range(njets):

                # Jet is defined by its constituents.
                jet_refs = jet_constituents[evt][i_jet].refs

                # Constituent indices in the ParticleFlowCandidate/ParticleFlowCandidate.fUniqueID array.
                indices = [id_to_index[id_value] for id_value in jet_refs]

                # Get the constituents' properties for the current jet.
                for v in constituent_properties.values():
                    event_store[v].append(list(constituent_properties_arrays[v][evt][indices]))
        
        # Append the event store to the main store.
        for v in constituent_properties.values():
            store[v].append(event_store[v])


    logger.info("still ok 1")
    # Convert each list in store to the awkward array.
    for v in constituent_properties.values():
        store[v] = ak.Array(store[v])
    logger.info("still ok 2")

    # Convert each awkward array to a tensor.
    tensors = {v: awkward_to_tensor(array) for v, array in store.items()}
    logger.info("still ok 3")

    # Create a TensorFlow dataset from the tensors.
    dataset = tf.data.Dataset.from_tensor_slices(tensors)
    logger.info("still ok 4")

    # Save the dataset.
    dataset.save(dataset_dir, compression='GZIP')
    logger.info("still ok 5")

    # Save the element_spec.
    with open(os.path.join(dataset_dir, 'element_spec.pkl'), 'wb') as f:
        pickle.dump(dataset.element_spec, f)
    logger.info("still ok 6")

    # Read the metadata from the text file created by the generate function and store it in a pickle file.
    metadata = qgpy.convert.read_metadata(f"{job_dir}/generate_metadata.txt")
    with open(os.path.join(dataset_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    logger.info("still ok 7")
    
    # Return.
    return

def read_metadata(metadata_file: str) -> Dict[str, Any]:
    """
    Read metadata from the text file.

    Parameters
    ----------
    metadata_file : str
        The path to the text file.

    Returns
    -------
    metadata : Dict[str, tf.constant]
        A dictionary containing the metadata.
    """

    logger = logging.getLogger("qgpy")
    logger.info(f"Getting metadata from the file: {metadata_file}")

    # Open the file and read the metadata.
    with open(metadata_file, 'r') as file:
        lines = file.readlines()
        labels, values = [], []
        for line in lines:
            key, value = line.strip().split(': ')
            labels.append(key)
            values.append(float(value))
        metadata = dict(zip(labels, tf.constant(values)))
        logger.info(f"Metadata: {metadata}")
    
    # Return the metadata.
    logger.info("Metadata read successfully.")
    return metadata
