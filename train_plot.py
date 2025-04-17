import matplotlib.pyplot as plt
from evaluation import evaluate
from IPython.display import clear_output


# Callback we associate with the StartEvent, sets up our new data.
def metric_start_plot():
    global train_rmse, train_mae, train_mape
    global val_rmse, val_mae, val_mape
    global epochs

    train_rmse, train_mae, train_mape = [], [], []
    val_rmse, val_mae, val_mape = [], [], []
    epochs = []


# Callback we associate with the EndEvent, do cleanup of data and figure.
def metric_end_plot():
    global train_rmse, train_mae, train_mape
    global val_rmse, val_mae, val_mape
    global epochs

    del train_rmse
    del train_mae 
    del train_mape
    del val_rmse
    del val_mae
    del val_mape
    del epochs
    # Close figure, we don't want to get a duplicate of the plot latter on
    plt.close()



def metric_plot_values(model, train_data, val_data, seq_len, scaler, epoch):
    global train_rmse, train_mae, train_mape
    global val_rmse, val_mae, val_mape
    global epochs
 
    train_metrics = evaluate(model, train_data, None, seq_len, scaler)
    val_metrics = evaluate(model, val_data, train_data, seq_len, scaler)
    
    train_rmse.append(train_metrics[0])
    train_mae.append(train_metrics[1])
    train_mape.append(train_metrics[2])
    val_rmse.append(val_metrics[0])
    val_mae.append(val_metrics[1])
    val_mape.append(val_metrics[2])
    epochs.append(epoch)

    # Clear the output area (wait=True, to reduce flickering), and plot
    # current data.

    clear_output(wait=True)
    # Plot the similarity metric values.
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    
    axs[0].plot(epochs, train_rmse, 'r-', label='Training')
    axs[0].plot(epochs, val_rmse, 'b-', label='Validation')
    axs[0].set_title('RMSE over Epochs')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('RMSE')
    axs[0].grid(True)
    axs[0].legend()

    # Subplot pour la métrique
    axs[1].plot(epochs, train_mae, 'r-', label='Training')
    axs[1].plot(epochs, val_mae, 'b-', label='Validation')
    axs[1].set_title('MAE over Epochs')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('MAE')
    axs[1].grid(True)
    axs[1].legend()
    
    # Subplot pour la métrique
    axs[2].plot(epochs, train_mape, 'r-', label='Training')
    axs[2].plot(epochs, val_mape, 'b-', label='Validation')
    axs[2].set_title('MAPE over Epochs')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('MAPE')
    axs[2].grid(True)
    axs[2].legend()
    
    plt.tight_layout()
    plt.show()
    
    

