import awkward as ak
import numpy as np

# A sample ak.Array of parton jet phi coordinates, now in [-pi, pi] range
parton_phis = ak.Array([
    np.pi * 0.9,     # Close to pi
    -np.pi * 0.1,    # Small negative value
    np.pi * 0.4,     # Small positive value
    -np.pi * 0.95,   # Close to -pi
])

# The reconstructed jet phi coordinate
reco_phi = np.pi * 0.99  # A value very close to -pi

# 1. Calculate the periodic difference
delta_phi = np.arctan2(
    np.sin(parton_phis - reco_phi),
    np.cos(parton_phis - reco_phi)
)

# 2. Take the absolute value to find the magnitude of the difference
abs_delta_phi = np.abs(delta_phi)

# 3. Find the index of the minimum difference
closest_jet_index = ak.argmin(abs_delta_phi, axis=None)

# --- Verification ---
print("Parton jet phis:")
print(parton_phis)
print(f"\nReconstructed jet phi: {reco_phi:.3f}")
print(f"Calculated periodic delta_phis: {abs_delta_phi}")

print("-" * 30)
print(f"Index of the closest parton jet: {closest_jet_index}")
print(f"Phi of the closest jet: {parton_phis[closest_jet_index]:.3f}")
print(f"Smallest delta_phi: {abs_delta_phi[closest_jet_index]:.3f}")