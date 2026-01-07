import sys
import os
import yaml
import torch
from pathlib import Path


# Define the path to the mad-lab root relative to this script
MAD_LAB_PATH = Path("/work/lei/mad-lab") 

if str(MAD_LAB_PATH) not in sys.path:
    print(f"Adding {MAD_LAB_PATH} to sys.path")
    sys.path.append(str(MAD_LAB_PATH))

try:
    from mad.configs import MADConfig
    from mad.data import generate_data
    from mad.registry import task_registry
except ImportError as e:
    print(f"Error importing MAD modules: {e}")
    print("Please verify MAD_LAB_PATH points to the correct root of the mad-lab repo.")
    sys.exit(1)

# Wrapper function
def get_mad_task_data(task_name: str, config_path: str = None, num_samples: int = 2):
    """
    Generates data for a MAD task, optionally overriding parameters with a YAML config.
    """
    print(f"\n=== Initializing Task: {task_name} ===")
    
    # 1. Initialize Default Config
    try:
        mad_config = MADConfig(task=task_name)
    except Exception as e:
        print(f"Failed to init MADConfig for {task_name}. valid tasks: {list(task_registry.keys())}")
        raise e

    current_kwargs = mad_config.instance_fn_kwargs.copy()
    print(f'sanity inspection (instance_fn): {mad_config.instance_fn}')

    # Apply YAML Overrides (if provided)
    if config_path and os.path.exists(config_path):
        print(f"Loading config overrides from: {config_path}")
        with open(config_path, 'r') as f:
            custom_cfg = yaml.safe_load(f)
        
        if custom_cfg:
            for k, v in custom_cfg.items():
                if k in current_kwargs:
                    print(f"  -> Overriding {k}: {current_kwargs[k]} -> {v}")
                else:
                    print(f"  -> Adding new param {k}: {v}")
                
                # Update our local dictionary
                current_kwargs[k] = v

    # Generate Data
    # use a local cache in this directory to avoid polluting the main repo
    local_cache = Path("./.data_cache") / task_name
    
    print("Generating/Loading data...")
    data = generate_data(
        instance_fn=mad_config.instance_fn,
        instance_fn_kwargs=current_kwargs,  
        num_train_examples=num_samples,
        num_test_examples=num_samples,
        train_data_path=str(local_cache / 'train'),
        test_data_path=str(local_cache / 'test'),
        num_workers=0 
    )
    
    return data

# TEST EXECUTION 

if __name__ == "__main__":
    # Example Usage
    target_task = 'noisy-in-context-recall' #'fuzzy-in-context-recall' 'selective-copying' 'compression' 'in-context-recall'
    
    # Optional: Create a dummy config file to test the override
    dummy_config = "my_test_config.yml"
    with open(dummy_config, 'w') as f:
        yaml.dump({'vocab_size': 16, 
                    # 'num_tokens_to_copy': 15, # specific param for selective copying
                    'k_motif_size': 3, # specific param for fuzzy in context recall
                    'v_motif_size': 6, # specific param for fuzzy in context recall
                    'seq_len': 64,
                    'num_noise_tokens': 1 # specific param for compression/fuzzy-in-context-recall
                    }, f)
    
    try:
        # Run the wrapper
        data_container = get_mad_task_data(target_task, config_path=dummy_config)
        
        # Inspect
        test_data = data_container['test']
        print(f"\nSuccessfully generated {len(test_data)} samples.")
        
        inputs, targets = next(iter(test_data))
        print(f"Sample Input Shape: {inputs.shape}")
        print(f"Sample Input Data:\n{inputs}")
        print(f"Sample Target Data:\n{targets}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup dummy config
        if os.path.exists(dummy_config):
            os.remove(dummy_config)