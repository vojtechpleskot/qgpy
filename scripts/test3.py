import pandas as pd
import awkward as ak

# Use the same dummy file as before
file_content = """\
event_id\tpt\tE\teta\tphi\tlabel1\tlabel2
1\t50.1\t60.2\t1.2\t-0.5\t1\t0
1\t25.3\t30.5\t-0.8\t1.1\t0\t1
2\t88.7\t90.0\t0.5\t2.0\t1\t1
2\t35.6\t40.1\t-0.1\t-1.5\t1\t0
2\t20.2\t22.4\t1.5\t0.3\t0\t0
3\t100.5\t102.8\t0.1\t-0.8\t1\t1
4\t9.8\t10.1\t-1.2\t-2.5\t0\t0
"""
with open("jet_data.txt", "w") as f:
    f.write(file_content)


def read_parton_jets(file_path):
    """
    Reads jet data using pandas and converts it to a dictionary
    of awkward arrays.

    Args:
        file_path (str): The path to the input text file.

    Returns:
        dict: A dictionary of awkward arrays.
    """
    # Use pandas to read the entire file into a flat DataFrame
    df = pd.read_csv(file_path, sep='\t', skiprows=1, header=None,
                     names=["event_number", "pt", "E", "eta", "phi", "IFN_label", "ATLAS_label"])

    # Get the number of jets per event, which determines the jaggedness
    counts = df.groupby('event_number').size().to_numpy()

    # Create the final dictionary for the awkward arrays
    awkward_arrays = {}
    
    # Iterate through the columns and convert each to a ragged awkward array
    for col in ["pt", "E", "eta", "phi", "IFN_label", "ATLAS_label"]:
        # Use ak.unflatten to directly create a ragged array from the flat data
        # and the counts for each event.
        awkward_arrays[col] = ak.unflatten(df[col].to_numpy(), counts)

    return awkward_arrays

# Run the function
# jet_data_arrays = read_jet_data_with_pandas("jet_data.txt")
jet_data_arrays = read_parton_jets("/scratch/ucjf-atlas/plesv6am/qg/data/slice0_0/generate.txt")


# You get the same result as before, but the process is much faster
# for large files.
print("Jet transverse momenta (pt):")
print(jet_data_arrays['pt'][13])
print(jet_data_arrays['E'][13])
print(jet_data_arrays['eta'][13])
print(jet_data_arrays['phi'][13])
print(jet_data_arrays['IFN_label'][13])
print(jet_data_arrays['ATLAS_label'][13])

print("\nNumber of jets per event:")
print(ak.num(jet_data_arrays['pt']))