import numpy as np
import torch
import matplotlib.pyplot as plt

def rmse(pred, target):
    return np.sqrt(((pred - target)**2 ).sum() / len(target))
    
def mae(pred, target):
    return (np.abs(pred - target)).sum() / len(target)

def mape(pred, target):
    return (np.abs((pred - target) / target)).sum() / len(target)


def predic_timeserie(model, data, past_data, seq_len):
    device = "cuda" if next(model.parameters()).is_cuda else "cpu"
    prediction = []
    seq = past_data[-seq_len:]
    for i in range(len(data)):
        torch_seq = torch.from_numpy(seq).unsqueeze(0).unsqueeze(-1).to(device)
        pred_t = model(torch_seq).item()

        seq = np.concatenate((seq[1:], [data[i]]), dtype=np.float32)
        prediction.append(pred_t)
    return prediction
        
        
def evaluate(model, data, past_data, seq_len, scaler):
    if past_data is None:
        past_data = data[:seq_len]
        data = data[seq_len:]
        
    prediction = predic_timeserie(model, data, past_data, seq_len)
    
    # Return to orignial scale for evaluation
    if scaler is not None: 
        data = scaler.inverse_transform(data.reshape(-1,1)).reshape(-1)
        prediction = scaler.inverse_transform(np.array(prediction).reshape(-1,1)).reshape(-1)
    
    return rmse(prediction, data), mae(prediction, data), mape(prediction, data)
    

def evaluate_naive(data, past_data, seq_len, scaler):
    if past_data is None:
        data = data[seq_len:]
        
    if scaler is not None: 
        data = scaler.inverse_transform(data.reshape(-1,1)).reshape(-1)
            
    prediction = np.concatenate(([data[0]], data[:-1]))
    
    return rmse(prediction, data), mae(prediction, data), mape(prediction, data)



def plot_prediction(model, data, past_data, seq_len, scaler):
    if past_data is None:
        past_data = data[:seq_len]
        data = data[seq_len:]
        
    prediction = predic_timeserie(model, data, past_data, seq_len)
    
    # Return to orignial scale for evaluation
    if scaler is not None: 
        data = scaler.inverse_transform(data.reshape(-1,1)).reshape(-1)
        prediction = scaler.inverse_transform(np.array(prediction).reshape(-1,1)).reshape(-1)
        
    fig, ax = plt.subplots(1,1)
    ax.plot(prediction, label="Prediction")
    ax.plot(data, label="True value")
    plt.legend(loc="best")
    