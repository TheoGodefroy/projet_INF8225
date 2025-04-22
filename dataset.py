from torch.utils.data import Dataset
import torch
import pandas as pd 
import numpy as np
from statsmodels.tsa.seasonal import STL

class TrainDataset(Dataset):
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
        


def extract_data(dataset, pays, prop, type, saison = None, ville = None, scaler=None):
    
    if sum(prop) != 1.:
        raise ValueError("The proportion must sum to 1") 
    prop = np.cumsum(prop)
    
    if type == 'chomage' : 
        print('Type : chomage')
        
        # Extract country 
        data = dataset[dataset["REF_AREA"] == pays].copy()
        # Transform date to datetime for sorting
        data.loc[:, 'TIME_PERIOD'] = pd.to_datetime(data["TIME_PERIOD"])
        data = data.sort_values(by='TIME_PERIOD')
        
        # Extract on unemployement rate
        data = np.array(data["OBS_VALUE"], dtype=np.float32)

    elif type == 'temperature' :
        print('Type : temperature')
        
        # On transforme les données pour qu'elles soient des moyennes mensuelle plutôt que des moyennes quotidiennes
        data = dataset.copy()
        data = data.groupby(['Country', 'City', 'Year', 'Month'])['AvgTemperature'].mean().reset_index()

        # On ajoute une colonne pour la date en mois de la température moyenne
        data['Time Period'] = data['Year'].astype(str) + '-' + data['Month'].astype(str)
        data.loc[:, 'Time Period'] = pd.to_datetime(data['Time Period']).dt.date

        # On garde seulement les valeurs pour le pays et la ville souhaitée
        data = data[(data["City"] == ville) & (data["Country"] == pays)].reset_index(drop=True)
        # On mets les données en ordre chronologique
        data = data.sort_values(by='Time Period')
        
        if saison == False : 
            # Comme les données du chômage étaient désaisonnalisées, on effectue le même traitement pour ces données 
            # qui ont clairement une saisonnalité sur les 12 mois de l'année

            stl = STL(data['AvgTemperature'], period = 12, robust = True)
            res = stl.fit()

            data['Saisonnalité'] = res.seasonal
            data['AvgTemperature Adjusted'] = data['AvgTemperature'] - data['Saisonnalité']
            
            data = np.array(data['AvgTemperature Adjusted'], dtype=np.float32)
        
        elif saison == True : 
            
            data = np.array(data['AvgTemperature'], dtype=np.float32)
        
    else : 
        print('Type de données non valide')
        return
        
    # Split to train/test sequence
    train_data = data[:int(prop[0] * len(data))]
    val_data =  data[int(prop[0] * len(data)): int(prop[1] * len(data))]
    test_data =  data[int(prop[1] * len(data)):]
    
    if scaler is not None:
        train_data = scaler.fit_transform(train_data.reshape(-1, 1)).reshape(-1)
        val_data = scaler.transform(val_data.reshape(-1, 1)).reshape(-1)
        test_data = scaler.transform(test_data.reshape(-1, 1)).reshape(-1)
    
    return train_data, val_data, test_data