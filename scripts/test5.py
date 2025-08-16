import pandas as pd
import numpy as np

# Create a sample DataFrame to replicate the user's data
# Values are in the [0, 2*pi] range
data = {
    'phi': [np.pi * 0.1, np.pi * 0.5, np.pi * 0.9, np.pi * 1.0, 
            np.pi * 1.1, np.pi * 1.5, np.pi * 1.9, np.pi * 2.0]
}
df = pd.DataFrame(data)

# The most efficient way to convert is with a vectorized operation
# using the modulo operator.
# This formula correctly wraps values from [pi, 2*pi] to [-pi, 0]
# and leaves values in [0, pi] unchanged.
df['phi_new'] = (df['phi'] + np.pi) % (2 * np.pi) - np.pi

# Print the original and new DataFrames to show the conversion
print("Original DataFrame:")
print(df[['phi']])

print("\nDataFrame with phi values in [-pi, pi] range:")
print(df[['phi_new']])