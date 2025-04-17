"""Add meta data to REINVENT model files"""
import sys
import uuid
import pickle
import time
import copy

import torch
import xxhash

from reinvent.models import meta_data

HASH_VERSION = f"xxhash.xxh3_128_hex {xxhash.VERSION}"


def update_model_data(save_dict: dict) -> dict:
    """Compute the hash for the model data

    Works on a copy of save_dict.

    :param save_dict: the model description
    :returns: updated save dict with the metadata as dict
    """

    # copy and sort
    save_dict = {k: v for k, v in sorted(save_dict.items())}

    metadata = save_dict["metadata"]
    if not isinstance(save_dict["metadata"], dict):
        metadata = metadata.as_dict()

    # do not hash the hash itself and its format
    metadata["hash_id"] = None
    metadata["hash_id_format"] = None

    # FIXME: what if this gets "too long"?
    metadata["updates"].append(time.time())

    ref = _get_network(save_dict)
    network = copy.deepcopy(ref)  # keep original tensors

    # convert to numpy arrays to avoid hashing on torch.tensor metadata
    # only needed for hashing, will copy back tensors further down
    for k in sorted(ref.keys()):
        ref[k] = ref[k].cpu().numpy()

    # FIX ! Need to hash the metadata dict not the object :)
    save_dict["metadata"] = metadata

    data = pickle.dumps(save_dict)

    metadata["hash_id"] = xxhash.xxh3_128_hexdigest(data)
    metadata["hash_id_format"] = HASH_VERSION

    return _set_network(save_dict, network)


def check_valid_hash(save_dict: dict) -> bool:
    """Check the hash of the model data

    Works on a copy of save_dict.  save_dict should not be used any further
    because the parameters, etc. are in numpy format.

    :param save_dict: the model description, metadata expected as dict
    :returns: whether hash is valid
    """

    save_dict = {k: v for k, v in sorted(save_dict.items())}

    metadata = save_dict["metadata"]

    curr_hash_id = metadata["hash_id"]
    curr_hash_id_format = metadata["hash_id_format"]

    # do not hash the hash and its format itself
    metadata["hash_id"] = None
    metadata["hash_id_format"] = None

    ref = _get_network(save_dict)

    for k in sorted(ref.keys()):
        ref[k] = ref[k].cpu().numpy()

    data = pickle.dumps(save_dict)

    check_hash_id = xxhash.xxh3_128_hexdigest(data)
    return curr_hash_id == check_hash_id


def _get_network(save_dict: dict) -> dict:
    if "decorator" in save_dict:  # Libinvent
        ref = save_dict["decorator"]["state"]
    elif "network_state" in save_dict:  # Linkinvnet, Mol2Mol
        ref = save_dict["network_state"]
    else:  # Reinvent
        ref = save_dict["network"]

    return ref


def _set_network(save_dict, network):
    if "decorator" in save_dict:  # Libinvent
        save_dict["decorator"]["state"] = network
    elif "network_state" in save_dict:  # Linkinvnet, Mol2Mol
        save_dict["network_state"] = network
    else:  # Reinvnet
        save_dict["network"] = network

    return save_dict


device = torch.device("cpu")
model = torch.load(sys.argv[1], map_location=device)

if "metadata" not in model:
    metadata = meta_data.ModelMetaData(
        hash_id=None,
        hash_id_format=HASH_VERSION,
        model_id=uuid.uuid4().hex,
        origina_data_source=sys.argv[3],
        creation_date=0,
    )

    model["metadata"] = metadata

new_model = update_model_data(model)
torch.save(new_model, sys.argv[2])

model = torch.load(sys.argv[2], map_location=device)

valid = check_valid_hash(model)
print(valid)
