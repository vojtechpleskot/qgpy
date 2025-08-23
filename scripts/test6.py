import tensorflow as tf

def create_simple_dataset(data, name):
    """
    Creates a simple tf.data.Dataset from a list of elements.
    Each element will be a dictionary with 'value' and 'source' keys.
    """
    def generator():
        for item in data:
            yield {"value": item, "source": name}
    
    # Use from_generator for more complex structures if needed, or from_tensor_slices for simpler ones
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_types={"value": tf.int32, "source": tf.string},
        output_shapes={"value": [], "source": []}
    )
    return dataset

# --- 1. Create several small tf.data.Dataset instances ---
print("Creating small datasets...")

dataset1_data = [1, 2, 3]
dataset2_data = [10, 11, 12, 13]
dataset3_data = [100, 101]

dataset1 = create_simple_dataset(dataset1_data, "Dataset-A")
dataset2 = create_simple_dataset(dataset2_data, "Dataset-B")
dataset3 = create_simple_dataset(dataset3_data, "Dataset-C")

print(f"Dataset 1 (elements: {len(dataset1_data)}): {list(dataset1.as_numpy_iterator())}")
print(f"Dataset 2 (elements: {len(dataset2_data)}): {list(dataset2.as_numpy_iterator())}")
print(f"Dataset 3 (elements: {len(dataset3_data)}): {list(dataset3.as_numpy_iterator())}\n")


# --- 2. Method 1: Concatenate datasets (sequential merging) ---
# This method appends one dataset after another.
print("--- Merging using concatenate() ---")
merged_dataset_concat = dataset1.concatenate(dataset2).concatenate(dataset3)

print("Concatenated Dataset Elements:")
for element in merged_dataset_concat:
    # Decode byte strings for readability
    value = element["value"].numpy()
    source = element["source"].numpy().decode('utf-8')
    print(f"  Value: {value}, Source: {source}")

print("\n--- Method 2: Interleave datasets (mixing elements) ---")
# This method allows you to interleave elements from multiple datasets.
# It's useful for shuffling or when you want to process data from different sources
# concurrently without waiting for one to finish entirely.

# `cycle_length`: Number of input elements that will be processed concurrently.
# `block_length`: Number of elements to take from each dataset before moving to the next.

# Let's create a list of datasets for interleave
all_datasets = [dataset1, dataset2, dataset3]

# Using from_tensor_slices to create a dataset of datasets
# (This is a common pattern for interleave)
dataset_of_datasets = tf.data.Dataset.from_tensor_slices(all_datasets)

# Now, interleave them
# Here, cycle_length=3 means it will try to get from up to 3 datasets at a time.
# block_length=1 means it takes 1 element from a dataset, then moves to the next
# available dataset, and so on.
merged_dataset_interleave = dataset_of_datasets.interleave(
    lambda x: x,
    cycle_length=3,
    block_length=1,
    num_parallel_calls=tf.data.AUTOTUNE # Allows TensorFlow to optimize parallelism
)

print("Interleaved Dataset Elements (cycle_length=3, block_length=1):")
for element in merged_dataset_interleave:
    value = element["value"].numpy()
    source = element["source"].numpy().decode('utf-8')
    print(f"  Value: {value}, Source: {source}")

# Example with different interleave parameters (e.g., taking more elements at once)
print("\nInterleaved Dataset Elements (cycle_length=2, block_length=2):")
merged_dataset_interleave_2 = dataset_of_datasets.interleave(
    lambda x: x,
    cycle_length=2, # Process up to 2 datasets concurrently
    block_length=2, # Take 2 elements from each dataset before switching
    num_parallel_calls=tf.data.AUTOTUNE
)
for element in merged_dataset_interleave_2:
    value = element["value"].numpy()
    source = element["source"].numpy().decode('utf-8')
    print(f"  Value: {value}, Source: {source}")


print("\n--- Considerations ---")
print("1. Concatenate: Simple sequential appending. Preserves order within original datasets.")
print("2. Interleave: Provides more control over how elements are mixed. Useful for shuffling,")
print("   or when datasets are large and you want to process them in parallel.")
print("3. `num_parallel_calls=tf.data.AUTOTUNE` is recommended for performance with `interleave`.")
print("4. All datasets being merged should have compatible output_types and output_shapes.")

