from qgpy.generate import generate_pythia

# Prepare a testing dictionary with the configuration.
test_cfg = {
    "executable": "cpp/generate",
    "nevents_per_job": 50,
    "seed": 0,
    "pTHatMin": 1000.0,
    "pTHatMax": 1500.0,
    "reco_jet_pt_min": 10.0,
    "log_level": "INFO"
}

# Call the generate_pythia function with the test configuration.
generate_pythia("/scratch/ucjf-atlas/plesv6am/qg/data/20250808/", test_cfg)
