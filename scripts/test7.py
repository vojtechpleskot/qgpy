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

# Get a list of all dataset directories
file_paths = [os.path.join(d, "tf_dataset") for d in glob("/scratch/ucjf-atlas/plesv6am/qg/data/slice0_*")]
print(f'\nFound {len(file_paths)} datasets on disk.')





# Dynamically read the output signature from the first dataset
first_dataset_path = file_paths[0]
temp_dataset = tf.data.Dataset.load(first_dataset_path)

output_signature = temp_dataset.element_spec
print("\nDynamically read output signature:")
print(output_signature)

# Create a Python generator function with error handling
def dataset_generator():
    """
    A Python generator that loads and yields elements from all datasets,
    skipping any that are corrupted.
    """
    for path in file_paths:
        try:
            dataset = tf.data.Dataset.load(path)
            # Iterate over the dataset to yield elements
            for element in dataset:
                yield element
        except tf.errors.DataLossError as e:
            # Catch the specific error and log a warning
            print(f"WARNING: Skipping corrupted dataset at path: {path}. Error: {e}")
        except Exception as e:
            # Catch any other unexpected errors
            print(f"WARNING: Skipping dataset at path: {path} due to unexpected error: {e}")

# Create the final dataset from the generator
concatenated_dataset = tf.data.Dataset.from_generator(
    generator=dataset_generator,
    output_signature=output_signature
)

# Verify the number of elements in the final dataset
try:
    total_elements = 0
    for _ in concatenated_dataset:
        total_elements += 1
    print(f'\nTotal elements in the concatenated dataset: {total_elements}')
except tf.errors.DataLossError as e:
    print(f"\nFATAL ERROR: A data loss error occurred while verifying the concatenated dataset. Check the source files.")
    print(e)

# If the verification succeeded, save the final concatenated dataset
if total_elements > 0:
    output_dir = './final_concatenated_dataset/'
    tf.data.Dataset.save(concatenated_dataset, output_dir)
    print(f'Successfully saved the final concatenated dataset to: {output_dir}')
