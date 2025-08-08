from omegaconf import OmegaConf

# Create a nested DictConfig object, similar to what Hydra would provide
cfg = OmegaConf.create(
    {
        "model": {
            "name": "MyModel",
            "layers": [
                {"type": "conv"},
                {"type": "pool"},
            ],
        },
        "training": {
            "epochs": 10,
            "learning_rate": 0.001,
        },
        # Example with a string interpolation
        "output_dir": "${env:HOME}/results/${model.name}"
    }
)

print("Original DictConfig Object:")
print(cfg)
print(f"Type: {type(cfg)}")

# 1. Convert to a dictionary without resolving interpolations
# The output_dir value will still be the unresolved string
my_dict_unresolved = OmegaConf.to_container(cfg, resolve=False)

print("\nConverted Dictionary (unresolved):")
print(my_dict_unresolved)
print(f"Type: {type(my_dict_unresolved)}")

# 2. Convert to a dictionary and resolve interpolations
# The output_dir value will be a fully resolved string
# Note: The output will depend on your HOME environment variable
my_dict_resolved = OmegaConf.to_container(cfg, resolve=True)

print("\nConverted Dictionary (resolved):")
print(my_dict_resolved)
