import numpy as np
import torch
from torch_geometric.data import Data

def load_dgraphfin_data(file_path='dgraphfin.npz'):
    """Load dgraphfin dataset from npz file"""
    data = np.load(file_path)
    
    # Convert to torch tensors
    x = torch.from_numpy(data['x']).float()
    y = torch.from_numpy(data['y']).long()
    edge_index = torch.from_numpy(data['edge_index']).long().t().contiguous()
    
    # Convert multi-class labels to binary (0=normal, 1=anomaly)
    # Assuming 0 is normal, others are anomalies
    y_binary = (y > 0).long()
    
    # Create PyG data object
    pyg_data = Data(x=x, edge_index=edge_index, y=y_binary)
    
    return pyg_data

def load_dgraph_data():
    """Load dgraph dataset - try pygod first, fallback to custom loading if needed"""
    try:
        from pygod.utils import load_data
        data = load_data('dgraph')
        return data
    except Exception as e:
        # If pygod fails, try to load from local npz file if exists
        try:
            import os
            if os.path.exists('dgraph.npz'):
                return load_dgraphfin_data('dgraph.npz')  # Reuse the same loading function
            else:
                raise RuntimeError(f"dgraph dataset not available. pygod error: {e}. Please ensure it's downloaded or available locally as dgraph.npz.")
        except Exception as e2:
            raise RuntimeError(f"dgraph dataset not available. pygod error: {e}. Local file error: {e2}")
