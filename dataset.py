from torch.utils.data import Dataset
import torch
import pandas as pd 
import numpy as np

class UnemployRateDataset(Dataset):
    def __init__(self, data, seq_len, shift):
        super().__init__()
        self.data  = torch.from_numpy(data).unfold(0, seq_len + 1, shift)
        
    def __len__(self):
        """Return the number of examples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, index):
    
        X =self.data[index][:-1]
        y = self.data[index][-1]
            
        return X,y
        


def extract_data(dataset, pays, prop, scaler=None):
    
    if sum(prop) != 1.:
        raise ValueError("The proportion mus sum to 1") 
    prop = np.cumsum(prop)
    
    # Extract country 
    data = dataset[dataset["REF_AREA"] == pays].copy()
    # Transform date to datetime for sorting
    data.loc[:, 'TIME_PERIOD'] = pd.to_datetime(data["TIME_PERIOD"])
    data = data.sort_values(by='TIME_PERIOD')
    
    # Extract on unemployement rate
    data = np.array(data["OBS_VALUE"], dtype=np.float32)
    # Split to train/test sequence
    train_data = data[:int(prop[0] * len(data))]
    val_data =  data[int(prop[0] * len(data)): int(prop[1] * len(data))]
    test_data =  data[int(prop[1] * len(data)):]

    if scaler is not None:
        train_data = scaler.fit_transform(train_data.reshape(-1, 1)).reshape(-1)
        val_data = scaler.transform(val_data.reshape(-1, 1)).reshape(-1)
        test_data = scaler.transform(test_data.reshape(-1, 1)).reshape(-1)
    
    return train_data, val_data, test_data