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
import vector

def delphes_to_jidenn(delphes_file: str, jidenn_file: str) -> None:
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

    # Register the vector library with Awkward Array
    vector.register_awkward()    

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

        # # Extract the jet_sd<x> values.
        # jet_sd = tree["Jet/Jet.SoftDroppedP4[5]"].array()
        # # From the last dimension of the jet_sd array, extract the first element.
        # for i in [0, 1, 2, 3, 4]:
        #     print(type(jet_sd[..., i]))
        #     print(type(jet_sd[..., i].type))
        #     jets = jet_sd[..., i]
        #     store[f'jets_sdmass{i}'] = ak.where(ak.num(jets) > 0, ak.firsts(jets).m, None)

        # jet_nparticles = tree["Jet/Jet.NCharged"].array() + tree["Jet/Jet.NNeutrals"].array()
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
