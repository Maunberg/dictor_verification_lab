# The script downloads model's weights or loads checkpoint from path
# Requirement: wget running on a Linux system 

# Import of modules
import os
import subprocess
from collections import OrderedDict

import torch

def load_model(model, lines, save_path=None, reload=False, checkpoint_path=None):
    # Load model's weights
    # If checkpoint_path is provided, load directly from it
    # Otherwise, download weights from provided URLs
    #
    # If save_path is None, do not download/check save_path (used if checkpoint_path is specified).

    # If checkpoint_path is specified, load checkpoint from it
    if checkpoint_path is not None:
        if not os.path.exists(checkpoint_path):
            raise ValueError(f'Checkpoint file {checkpoint_path} does not exist.')
        checkpoint = torch.load(checkpoint_path)
    else:
        # If save_path is None and no checkpoint specified, we cannot download
        if save_path is None:
            raise ValueError("save_path must be specified if checkpoint_path is not provided.")

        # Create save_path directory if it does not exist
        if not os.path.exists(save_path):
            os.mkdir(save_path, mode=0o777)

        # By default, use 'baseline_v2_ap.model' as model filename
        model_filename = 'baseline_v2_ap.model'

        for line in lines:
            url     = line.strip()
            outfile = url.split('/')[-1]

            out = 0

            # Download files
            if not os.path.exists(os.path.join(save_path, outfile)) or reload:
                out = subprocess.call('wget %s -O %s/%s'%(url, save_path, outfile), shell=True)

            if out != 0:
                raise ValueError('Download failed %s. If download fails repeatedly, use alternate URL on the VoxCeleb website.' % url)

            print('File %s is downloaded.' % outfile)
            model_filename = outfile  # use the last downloaded filename

        checkpoint = torch.load(os.path.join(save_path, model_filename))

    # Determine actual model state dict location inside checkpoint.
    if isinstance(checkpoint, dict) and 'model' in checkpoint and isinstance(checkpoint['model'], dict):
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # Some checkpoints store weights with prefixes like '__S__' or 'module.'.
    # Normalize keys before loading into the model.
    model_weight = OrderedDict()
    has_special_prefix = any('__S__' in key for key in state_dict.keys())

    def _strip_prefix(key: str) -> str:
        if key.startswith('__S__.'):
            return key[6:]
        if key.startswith('__S__'):
            return key[5:]
        if key.startswith('module.'):
            return key[7:]
        return key

    if has_special_prefix:
        for key, value in state_dict.items():
            if '__S__' in key:
                cleaned_key = _strip_prefix(key)
                model_weight[cleaned_key] = value
    else:
        for key, value in state_dict.items():
            cleaned_key = _strip_prefix(key)
            model_weight[cleaned_key] = value

    model.load_state_dict(model_weight)

    return model