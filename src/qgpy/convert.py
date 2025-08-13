"""
convert.py
==========

Functions for converting between different file formats used in the analysis pipeline.

Functions
---------

"""

import uproot
import numpy as np
import awkward as ak
import qgpy.utils
from typing import Dict, Any
import logging
import tensorflow as tf

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

def delphes_to_jidenndataset(delphes_file: str, jidenn_dir: str) -> None:

    # Create a logger instance.
    outdir = delphes_file.rsplit('/', 1)[0]
    logger = qgpy.utils.create_logger('convert', outdir = outdir)
    logger.info("Starting the Delphes to JIDENN conversion...")

    # Extract the branches from the tree
    jet_constituents = tree["Jet/Jet.Constituents"].array()
    constituent_funique_ids = tree["ParticleFlowCandidate/ParticleFlowCandidate.fUniqueID"].array()
    constituents_properties_arrays = {
        prop : tree[prop].array() for prop in constituents_properties
    }
    print(jet_constituents)

    # Create an awkward array of jet constituent properties (pt, eta, phi, E, and charge),
    # belonging to each jet in the event.
    # The first dimension is the events, the second is the jets, and the third is the constituents.
    # Each jet has an array of references to the ParticleFlowCandidate objects; it is stored in the "refs" attribute within the Jet/Jet.Constituents branch.
    # The constituents' properties are stored in the ParticleFlowCandidate/ParticleFlowCandidate.PT/Eta/Phi/E/Charge branches.

    # Create an entry for each constituents' property and for each jet in the store dictionary.
    for prop in constituents_properties:
        for i_jet in range(njets):
            store[f"jet_{i_jet}.c_{prop.split('.')[-1]}"] = []

    # Number of events in the root file.
    n_events = len(jet_constituents)

    # Loop over all events in the root file.
    for evt in range(n_events):

        # Print the progress every 10 events.
        if evt % 10 == 0:
            print(f"Processing event {evt} of {n_events}")

        # Map the unique IDs of the constituents to their indices in the constituent_funique_ids array.
        id_to_index = {id_value: index for index, id_value in enumerate(constituent_funique_ids[evt])}

        # Loop over jets in the event.
        for i_jet in range(njets):

            # Jet is defined by its constituents.
            jet_refs = jet_constituents[evt][i_jet].refs

            # Constituent indices in the ParticleFlowCandidate/ParticleFlowCandidate.fUniqueID array.
            indices = [id_to_index[id_value] for id_value in jet_refs]

            # Get the constituents' properties for the current jet.
            for prop in constituents_properties:
                store[f"jet_{i_jet}.c_{prop.split('.')[-1]}"].append(list(constituents_properties_arrays[prop][evt][indices]))

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
