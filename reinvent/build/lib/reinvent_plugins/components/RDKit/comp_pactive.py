"""Group count

Allow groups of atoms defined by SMARTS only a certain number of times in a
molecules.
"""

from __future__ import annotations

__all__ = ["PActive"]

from typing import List

from rdkit import Chem

from rdkit import DataStructs
from rdkit.Chem import AllChem
import numpy as np
from pydantic.dataclasses import dataclass

from ..component_results import ComponentResults
from reinvent_plugins.mol_cache import molcache
from ..add_tag import add_tag

import pickle


class BinaryRandomForest:
    def __init__(self, states, max_feature_dim):
        """
        State is a list of dictionary
        """
        self.states = states
        self.max_feature_dim = max_feature_dim

    def predict_proba(self, X):
        X = np.array(X, dtype=np.float32)
        X = X[:, :self.max_feature_dim+1]

        preds = np.zeros((len(X), 2), dtype=np.float32)

        for i, x in enumerate(X):
            for state in self.states:
                preds[i : i + 1] += self.decision_tree_prediction(x, state)

        return preds / len(self.states)

    def get_node_feature(self, node):
        left, right, feature_id, th, impurity, n_node_samples, weighted_n_node_samples = node
        left = int(left)
        right = int(right)
        feature_id = int(feature_id)
        n_node_samples = int(n_node_samples)
        return left, right, feature_id, th, impurity, n_node_samples, weighted_n_node_samples

    @staticmethod
    def convert_sklearn_to_npy_checkpoint(sklearn_chkpt_path, npy_chkpt_path):
        with gzip.open(sklearn_chkpt_path) as f:
            model = pickle.load(f)
        
        states = []
        max_feature_dim = 0
        for dt in model.estimators_:
            tree = dt.tree_
            state = tree.__getstate__()
            max_feature_dim = max(max_feature_dim, max([x[2] for x in state["nodes"]]))
            state["nodes"] = np.array([np.array(x.tolist(), dtype=np.float32) for x in state["nodes"]])
            state["values"] = state["values"].astype(np.float32)
            states.append(state)
        
        with gzip.open(npy_chkpt_path, "wb") as f:
            pickle.dump({"states": states, "max_feature_dim":max_feature_dim}, f)
             
    def decision_tree_prediction(self, x, state):
        tree_nodes = state["nodes"]
        tree_values = state["values"]
        max_depth = state["max_depth"]

        feature_id = 0  # root node

        path = [feature_id]
        value = tree_values[feature_id]
        (
            left,
            right,
            feature_id,
            th,
            _,
            _,
            _,
        ) = self.get_node_feature(tree_nodes[feature_id])

        for depth in range(max_depth):
            if x[feature_id] <= th:
                feature_id = left
                path.append(left)
            else:
                feature_id = right
                path.append(right)

            value = tree_values[feature_id]
            (
                left,
                right,
                feature_id,
                th,
                _,
                _,
                _,
            ) = self.get_node_feature(tree_nodes[feature_id])

            if (left == -1) and (right == -1) and (feature_id == -2):  # leaf node
                break

        return value / value.sum()


@add_tag("__parameters")
@dataclass
class Parameters:
    """Parameters for the scoring component

    Note that all parameters are always lists because components can have
    multiple endpoints and so all the parameters from each endpoint is
    collected into a list.  This is also true in cases where there is only one
    endpoint.
    """
    pickle_path: List[str]


@add_tag("__component")
class PActive:
    def __init__(self, params: Parameters):
        self.number_of_endpoints = 1
        params = pickle.load(open(params.pickle_path[0], "rb"))
        states, max_feature_dim = params["states"], params["max_feature_dim"]
        self.model = BinaryRandomForest(states, max_feature_dim)

    @molcache
    def __call__(self, mols: List[Chem.Mol]) -> np.array:
        fps = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=3) for mol in mols]
        X = []
        for fp in fps:
            x = np.zeros(2048, dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(fp, x)
            X.append(x)
        X = np.array(X)
        scores = self.model.predict_proba(X)[:, 1]
        return ComponentResults([scores])