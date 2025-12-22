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
import pandas as pd

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
        store['jets_e'] = ak.Array(np.sqrt(total_momentum**2 + jet_mass**2))

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

def delphes_to_tf_dataset(job_dir: str, delphes_file: str, labels_file: str, dataset_dir: str) -> None:

    # Create a logger instance.
    outdir = delphes_file.rsplit('/', 1)[0]
    logger = qgpy.utils.create_logger('convert', outdir = outdir)
    logger.info("Starting the Delphes to JIDENN conversion...")
        
    # Read the parton jets from the text file.
    logger.info(f"Reading parton jets from the file: {labels_file}")
    parton_jets = read_parton_jets(labels_file)

    # Define constituent properties to extract (keys) and their corresponding awkward array names (values).
    constituent_properties = {
        "ParticleFlowCandidate/ParticleFlowCandidate.PT": "jets_c_pt",
        "ParticleFlowCandidate/ParticleFlowCandidate.Eta": "jets_c_eta",
        "ParticleFlowCandidate/ParticleFlowCandidate.Phi": "jets_c_phi",
        "ParticleFlowCandidate/ParticleFlowCandidate.E": "jets_c_e",
        "ParticleFlowCandidate/ParticleFlowCandidate.Mass": "jets_c_m",
        "ParticleFlowCandidate/ParticleFlowCandidate.Charge": "jets_c_charge",
    }

    # Create a dictionary to store the jet properties.
    store = {v: [] for v in list(constituent_properties.values()) + ['jets_ifn_label', 'jets_atlas_label']}

    # Open the ROOT file.
    logger.info(f"Opening the Delphes root file: {delphes_file}")
    with uproot.open(delphes_file) as file:

        # Access the TTree named "Delphes"
        tree = file["Delphes"]

        # List all keys in the root file
        keys = tree.keys()
        logger.debug("Keys in the root file:")
        for key in keys:
            logger.debug(f'{key}, {type(tree[key])}')

        # Read the global jet variables.
        logger.info("Reading global jet variables...")
        store['jets_pt']  = tree["Jet/Jet.PT"].array()
        store['jets_eta'] = tree["Jet/Jet.Eta"].array()
        store['jets_phi'] = tree["Jet/Jet.Phi"].array()
        store['jets_m']   = tree["Jet/Jet.Mass"].array()

        # Calculate the jet energy from pt, eta, phi, and mass.
        jet_mass = tree["Jet/Jet.Mass"].array()
        total_momentum = store['jets_pt'] * np.cosh(store['jets_eta'])
        store['jets_energy'] = ak.Array(np.sqrt(total_momentum**2 + jet_mass**2))

        # The flavor label evaluated by Delphes.        
        store['jets_delphes_label'] = ak.values_astype(tree["Jet/Jet.Flavor"].array(), 'int32')
        store['jets_PartonTruthLabelID'] = store['jets_delphes_label']

        # -----------------------------------------------------        
        # -----------------------------------------------------
        # ----- This is not really needed in this function

        # Extract the jet_tau<x> values.
        jet_tau = tree["Jet/Jet.Tau[5]"].array()
        # From the last dimension of the jet_tau array, extract the first element.
        for i in [1, 2, 3, 4]:
            store[f'jets_tau{i}'] = jet_tau[..., i]

        # Extract the number of particles in a jet.
        store['jets_nparticles'] = tree["Jet/Jet.NCharged"].array() + tree["Jet/Jet.NNeutrals"].array()

        # Copy the array store['jets_pt'] and fill the copy with zeros.
        store['jets_sdmass'] = ak.zeros_like(store['jets_pt'])

        # ----- End of the not needed part
        # -----------------------------------------------------        
        # -----------------------------------------------------        

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
        for i_evt in range(nevents):
            # Print the progress every 10 events.
            if i_evt % 100 == 0:
                logger.info(f"Processing event {i_evt} of {nevents}")

            # Create a dictionary to store the jet properties for the current event.
            event_store = {v: [] for v in list(constituent_properties.values()) + ['jets_ifn_label', 'jets_atlas_label']}

            # Map the unique IDs of the constituents to their indices in the constituent_funique_ids array.
            id_to_index = {id_value: index for index, id_value in enumerate(constituent_funique_ids[i_evt])}

            njets = len(jet_constituents[i_evt])
            logger.debug(f"Number of jets in the event {i_evt}: {njets}")

            # Loop over jets in the event.
            for i_jet in range(njets):

                # Jet is defined by its constituents.
                jet_refs = jet_constituents[i_evt][i_jet].refs

                # Constituent indices in the ParticleFlowCandidate/ParticleFlowCandidate.fUniqueID array.
                indices = [id_to_index[id_value] for id_value in jet_refs]

                # Get the constituents' properties for the current jet.
                for v in constituent_properties.values():
                    event_store[v].append(list(constituent_properties_arrays[v][i_evt][indices]))

                # Map the reco jet to a parton jet, based on the proximity in the eta-phi space.
                # The reco jet is matched to the closest parton jet if their Delta_R < 0.4.
                # Read the IFN and ATLAS labels from the matched parton jet and append them to the event store.
                # Use the -1 code to indicate that there is no matching parton jet for a given reco jet.
                parton_jet_index = dr_matching(
                    reco_jet_eta = store['jets_eta'][i_evt][i_jet],
                    reco_jet_phi = store['jets_phi'][i_evt][i_jet],
                    parton_jets_eta = parton_jets['eta'][i_evt],
                    parton_jets_phi = parton_jets['phi'][i_evt],
                )
                if parton_jet_index >= 0:
                    ifn_label = parton_jets['IFN_label'][i_evt][parton_jet_index]
                    atlas_label = parton_jets['ATLAS_label'][i_evt][parton_jet_index]
                    logger.debug(f"Event: {i_evt}, Reco jet: {i_jet}")
                    logger.debug(f"Matched parton jet index: {parton_jet_index}, IFN label: {ifn_label}, ATLAS label: {atlas_label}")
                    logger.debug(f"Parton jet eta: {parton_jets['eta'][i_evt][parton_jet_index]}, phi: {parton_jets['phi'][i_evt][parton_jet_index]}")
                    logger.debug(f"Reco jet eta: {store['jets_eta'][i_evt][i_jet]}, phi: {store['jets_phi'][i_evt][i_jet]}")
                    logger.debug(f"Parton jet pt: {parton_jets['pt'][i_evt][parton_jet_index]}")
                    logger.debug(f"Reco jet pt: {store['jets_pt'][i_evt][i_jet]}")

                else:
                    ifn_label = -1
                    atlas_label = -1
                event_store['jets_ifn_label'].append(ifn_label)
                event_store['jets_atlas_label'].append(atlas_label)
        
            # Append the event store to the main store.
            for v in list(constituent_properties.values()) + ['jets_ifn_label', 'jets_atlas_label']:
                store[v].append(event_store[v])


    # Convert each list in store to the awkward array.
    for v in list(constituent_properties.values()) + ['jets_ifn_label', 'jets_atlas_label']:
        store[v] = ak.Array(store[v])

    # Convert each awkward array to a tensor.
    tensors = {v: awkward_to_tensor(array) for v, array in store.items()}

    # Create a TensorFlow dataset from the tensors.
    dataset = tf.data.Dataset.from_tensor_slices(tensors)

    # Save the dataset.
    dataset.save(dataset_dir, compression='GZIP')

    # Save the element_spec.
    with open(os.path.join(dataset_dir, 'element_spec.pkl'), 'wb') as f:
        pickle.dump(dataset.element_spec, f)

    # Read the metadata from the text file created by the generate function and store it in a pickle file.
    metadata = read_metadata(f"{job_dir}/generate_metadata.txt")
    with open(os.path.join(dataset_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    
    # Return.
    return


def delphes_to_tf_dataset_global_vars(job_dir: str, delphes_file: str, labels_file: str, dataset_dir: str) -> None:

    # Create a logger instance.
    outdir = delphes_file.rsplit('/', 1)[0]
    logger = qgpy.utils.create_logger('convert', outdir = outdir)
    logger.info("Starting the Delphes to JIDENN conversion...")
        
    # Read the parton jets from the text file.
    logger.info(f"Reading parton jets from the file: {labels_file}")
    parton_jets = read_parton_jets(labels_file)

    # Create a dictionary to store the jet properties.
    store = {v: [] for v in ['jets_ifn_label', 'jets_atlas_label']}

    # Open the ROOT file.
    logger.info(f"Opening the Delphes root file: {delphes_file}")
    with uproot.open(delphes_file) as file:

        # Access the TTree named "Delphes"
        tree = file["Delphes"]

        # List all keys in the root file
        keys = tree.keys()
        logger.debug("Keys in the root file:")
        for key in keys:
            logger.debug(f'{key}, {type(tree[key])}')

        # Read the global jet variables.
        logger.info("Reading global jet variables...")
        store['jets_pt'] = tree["Jet/Jet.PT"].array()
        store['jets_eta'] = tree["Jet/Jet.Eta"].array()
        store['jets_phi'] = tree["Jet/Jet.Phi"].array()

        # Calculate the jet energy from pt, eta, phi, and mass.
        jet_mass = tree["Jet/Jet.Mass"].array()
        total_momentum = store['jets_pt'] * np.cosh(store['jets_eta'])
        store['jets_energy'] = ak.Array(np.sqrt(total_momentum**2 + jet_mass**2))

        # The flavor label evaluated by Delphes.        
        store['jets_delphes_label'] = ak.values_astype(tree["Jet/Jet.Flavor"].array(), 'int32')
        store['jets_PartonTruthLabelID'] = store['jets_delphes_label']

        # -----------------------------------------------------        
        # -----------------------------------------------------
        # ----- This is not really needed in this function

        # Extract the jet_tau<x> values.
        jet_tau = tree["Jet/Jet.Tau[5]"].array()
        # From the last dimension of the jet_tau array, extract the first element.
        for i in [1, 2, 3, 4]:
            store[f'jets_tau{i}'] = jet_tau[..., i]

        # Extract the number of particles in a jet.
        store['jets_nparticles'] = tree["Jet/Jet.NCharged"].array() + tree["Jet/Jet.NNeutrals"].array()

        # Copy the array store['jets_pt'] and fill the copy with zeros.
        store['jets_sdmass'] = ak.zeros_like(store['jets_pt'])

        # ----- End of the not needed part
        # -----------------------------------------------------        
        # -----------------------------------------------------        

        # Number of events in the root file.
        nevents = len(jet_mass)
        logger.info(f"Number of events in the root file: {nevents}")

        # Loop over all events in the root file.
        for i_evt in range(nevents):
            # Print the progress every 10 events.
            if i_evt % 100 == 0:
                logger.info(f"Processing event {i_evt} of {nevents}")

            # Create a dictionary to store the jet properties for the current event.
            event_store = {v: [] for v in ['jets_ifn_label', 'jets_atlas_label']}

            njets = len(jet_mass[i_evt])
            logger.debug(f"Number of jets in the event {i_evt}: {njets}")

            # Loop over jets in the event.
            for i_jet in range(njets):

                # Map the reco jet to a parton jet, based on the proximity in the eta-phi space.
                # The reco jet is matched to the closest parton jet if their Delta_R < 0.4.
                # Read the IFN and ATLAS labels from the matched parton jet and append them to the event store.
                # Use the -1 code to indicate that there is no matching parton jet for a given reco jet.
                parton_jet_index = dr_matching(
                    reco_jet_eta = store['jets_eta'][i_evt][i_jet],
                    reco_jet_phi = store['jets_phi'][i_evt][i_jet],
                    parton_jets_eta = parton_jets['eta'][i_evt],
                    parton_jets_phi = parton_jets['phi'][i_evt],
                )
                if parton_jet_index >= 0:
                    ifn_label = parton_jets['IFN_label'][i_evt][parton_jet_index]
                    atlas_label = parton_jets['ATLAS_label'][i_evt][parton_jet_index]
                    logger.debug(f"Event: {i_evt}, Reco jet: {i_jet}")
                    logger.debug(f"Matched parton jet index: {parton_jet_index}, IFN label: {ifn_label}, ATLAS label: {atlas_label}")
                    logger.debug(f"Parton jet eta: {parton_jets['eta'][i_evt][parton_jet_index]}, phi: {parton_jets['phi'][i_evt][parton_jet_index]}")
                    logger.debug(f"Reco jet eta: {store['jets_eta'][i_evt][i_jet]}, phi: {store['jets_phi'][i_evt][i_jet]}")
                    logger.debug(f"Parton jet pt: {parton_jets['pt'][i_evt][parton_jet_index]}")
                    logger.debug(f"Reco jet pt: {store['jets_pt'][i_evt][i_jet]}")

                else:
                    ifn_label = -1
                    atlas_label = -1
                event_store['jets_ifn_label'].append(ifn_label)
                event_store['jets_atlas_label'].append(atlas_label)
        
            # Append the event store to the main store.
            for v in ['jets_ifn_label', 'jets_atlas_label']:
                store[v].append(event_store[v])


    # Convert each list in store to the awkward array.
    for v in ['jets_ifn_label', 'jets_atlas_label']:
        store[v] = ak.Array(store[v])

    # Convert each awkward array to a tensor.
    # tensors = {v: awkward_to_tensor(array) for v, array in store.items()}
    tensors = {v: ak.to_raggedtensor(array) for v, array in store.items()}

    # Create a TensorFlow dataset from the tensors.
    dataset = tf.data.Dataset.from_tensor_slices(tensors)

    # Save the dataset.
    dataset.save(dataset_dir, compression='GZIP')

    # Save the element_spec.
    with open(os.path.join(dataset_dir, 'element_spec.pkl'), 'wb') as f:
        pickle.dump(dataset.element_spec, f)

    # Read the metadata from the text file created by the generate function and store it in a pickle file.
    metadata = read_metadata(f"{job_dir}/generate_metadata.txt")
    with open(os.path.join(dataset_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    
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

def read_parton_jets(text_file: str) -> Dict[str, ak.Array]:
    """
    Read the parton jet properties from the text file.

    Parameters
    ----------
    text_file : str
        The path to the text file containing jets and their IFN and ATLAS labels.

    Returns
    -------
    ret : Dict[str, ak.Array]
        A dictionary of awkward arrays containing the jets and their IFN and ATLAS labels.
    """

    # Get the logger and log the file being read.
    logger = logging.getLogger("qgpy")
    logger.info(f"Getting parton jet properties from the file: {text_file}")

    # Use pandas to read the entire file into a flat DataFrame
    df = pd.read_csv(text_file, sep=',', skiprows=1, header=None,
                     names=["event_number", "pt", "E", "eta", "phi", "IFN_label", "ATLAS_label"])

    # Convert the phi values to the range [-pi, pi] for consistency with the delphes output.
    # Note that Pythia generates phi in the range [0, 2*pi], so we need to adjust it.
    df['phi'] = (df['phi'] + np.pi) % (2 * np.pi) - np.pi

    # Get the number of jets per event, which determines the jaggedness
    counts = df.groupby('event_number').size().to_numpy()

    # Create the final dictionary for the awkward arrays
    awkward_arrays = {}

    # Iterate through the columns and convert each to a ragged awkward array
    for col in ["pt", "E", "eta", "phi", "IFN_label", "ATLAS_label"]:
        # Use ak.unflatten to directly create a ragged array from the flat data
        # and the counts for each event.
        awkward_arrays[col] = ak.unflatten(df[col].to_numpy(), counts)

    # Log the end of the function.
    logger.info("Parton jets read successfully.")

    # Return the labels.
    return awkward_arrays

def dr_matching(reco_jet_eta: float, reco_jet_phi: float, parton_jets_eta: ak.Array, parton_jets_phi: ak.Array) -> ak.Array:
    """
    Match the reco jet to the closest parton jet based on Delta_R.

    Parameters
    ----------
    reco_jet_eta : float
        The eta of the reco jet.
    reco_jet_phi : float
        The phi of the reco jet.
    parton_jets_eta : ak.Array
        The eta of the parton jets.
    parton_jets_phi : ak.Array
        The phi of the parton jets.

    Returns
    -------
    ret : int
        The index of the matched parton jet; -1 if no match is found.
    """

    # 1. Calculate the periodic dphi difference
    delta_phi = np.arctan2(
        np.sin(parton_jets_phi - reco_jet_phi),
        np.cos(parton_jets_phi - reco_jet_phi)
    )

    # 2. Calculate Delta_R for each parton jet
    delta_r = np.sqrt(
        (parton_jets_eta - reco_jet_eta) ** 2 + delta_phi ** 2
    )

    # 3. Find the index of the minimum difference
    closest_jet_index = ak.argmin(delta_r, axis=None)

    # 4. Check if the closest jet is within the Delta_R < 0.4
    if delta_r[closest_jet_index] < 0.4:
        matched_index = closest_jet_index
    else:
        matched_index = -1

    return matched_index
