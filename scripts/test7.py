from glob import glob
import os
import tensorflow as tf

# for i,folder in enumerate(glob("/scratch/ucjf-atlas/plesv6am/qg/data/slice0_*")):
#     print(f"Found dataset {i}: {folder}")
#     dataset = tf.data.Dataset.load(os.path.join(folder, "tf_dataset"))
#     if i == 0:
#         concatenated_dataset = dataset
#     else:
#         concatenated_dataset = concatenated_dataset.concatenate(dataset)

# # Store the concatenated dataset on the disk, using the experimental method.
# tf.data.experimental.save(concatenated_dataset, "/scratch/ucjf-atlas/plesv6am/qg/data/concatenated_dataset")

# files = [os.path.join(d, "tf_dataset") for d in glob("/scratch/ucjf-atlas/plesv6am/qg/data/slice0_*")]
# print(files)
# ds = tf.data.Dataset.from_tensor_slices(files)
# # print the number of events in ds
# print("Number of events in dataset:", ds.cardinality().numpy())
# ds.save("/scratch/ucjf-atlas/plesv6am/qg/data/concatenated_dataset", compression="GZIP")


import shutil

# --- Part 2: Read, Concatenate, and Store the Datasets ---
# Define the directory where your datasets are stored
data_dir = './my_datasets/'

# Get a list of all dataset directories
file_paths = [os.path.join(d, "tf_dataset") for d in glob("/scratch/ucjf-atlas/plesv6am/qg/data/slice0_*")]
print(f'\nFound {len(file_paths)} datasets on disk.')

# Create a dataset from the list of file paths
paths_dataset = tf.data.Dataset.from_tensor_slices(file_paths)

# This is the Python function that will be wrapped
def _load_dataset(path_tensor):
    # This part runs in Python, not in the graph
    python_string_path = path_tensor.numpy().decode('utf-8')
    return tf.data.Dataset.load(python_string_path)

# This is the wrapper function to be passed to interleave
def load_dataset_from_path(path_tensor):
    # Define the output signature of the wrapped function
    output_signature = tf.data.DatasetSpec(
        element_spec={
            'var_A': tf.TensorSpec(shape=(None, 5), dtype=tf.float32),
            'var_B': tf.TensorSpec(shape=(None,), dtype=tf.int32)
        }
    )
    
    # Use tf.py_function to wrap the Python function
    return tf.py_function(
        func=_load_dataset,
        inp=[path_tensor],
        Tout=tf.data.Dataset,
        output_signature=output_signature
    )


# Use `interleave` to read and concatenate the datasets in a streaming fashion
concatenated_dataset = paths_dataset.interleave(
    load_dataset_from_path,
    cycle_length=tf.data.AUTOTUNE,
    num_parallel_calls=tf.data.AUTOTUNE
)

# Optional: Add any further transformations here
# concatenated_dataset = concatenated_dataset.shuffle(buffer_size=5000)

# Verify the number of elements in the final dataset
total_elements = 0
for _ in concatenated_dataset:
    total_elements += 1
print(f'Total elements in the concatenated dataset: {total_elements}')
print(f'Expected elements: {len(file_paths) * 10}')

# Store the final concatenated dataset
output_dir = './final_concatenated_dataset/'
tf.data.experimental.save(concatenated_dataset, output_dir)
print(f'\nSuccessfully saved the final concatenated dataset to: {output_dir}')

# To load the final dataset later:
# final_dataset = tf.data.Dataset.load(output_dir)
# print(f'\nSuccessfully loaded the final dataset from: {output_dir}')