# Anomaly detection in electrocardiogram (ECG) data
We use PyTorch to build an unsupervised long short-term memory (LSTM) autoencoder to detect anomalies in time series data of ECG heartbeat waveforms.


```python
import os
import random

import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
%matplotlib inline
```


```python
# Check if CUDA is available and set variable 'device' accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### The dataset
The raw ECG data is available at the [PTB Diagnostic ECG Database](https://www.physionet.org/content/ptbdb/1.0.0/) and was collected from heart-disease patients and healthy volunteers. We use already preprocessed data which is segmented to individual heartbeats from [kaggle](https://www.kaggle.com/shayanfazeli/heartbeat). We note that the preprocessing steps are not outlined in detail and that the clinical summaries corresponding to the individual waveforms are not available any more. Thus, the performance of the anomaly detection will not be optimimal. However, we will focus on the implementation of the LSTM autoencoder and not on the applicability in the real world. 


```python
dataDir = 'data'

# The heartbeats are categorized in two categories: normal/healthy and abnormal
csvN = os.path.join(dataDir, 'ptbdb_normal.csv')
csvA = os.path.join(dataDir, 'ptbdb_abnormal.csv')

# Load dataset
dfN = pd.read_csv(csvN, header=None)
dfA = pd.read_csv(csvA, header=None)

# Drop the last column which seems to be the class labels (i.e., 0 or 1)
dfN.drop(dfN.columns[len(dfN.columns)-1], axis=1, inplace=True)
dfA.drop(dfA.columns[len(dfA.columns)-1], axis=1, inplace=True)

# Shuffle data
dfN = dfN.sample(frac=1., random_state=42).reset_index(drop=True)
dfA = dfA.sample(frac=1., random_state=42).reset_index(drop=True)
```


```python
# Each row corresponds to a cropped and downsampled heartbeat
# Each heartbeat has 188 normalized ECG values and
dfN
```




<div>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>177</th>
      <th>178</th>
      <th>179</th>
      <th>180</th>
      <th>181</th>
      <th>182</th>
      <th>183</th>
      <th>184</th>
      <th>185</th>
      <th>186</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.997871</td>
      <td>0.847595</td>
      <td>0.428693</td>
      <td>0.192422</td>
      <td>0.044274</td>
      <td>0.009366</td>
      <td>0.080034</td>
      <td>0.122180</td>
      <td>0.142614</td>
      <td>0.153682</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.000000</td>
      <td>0.454686</td>
      <td>0.067971</td>
      <td>0.000515</td>
      <td>0.000000</td>
      <td>0.083934</td>
      <td>0.126674</td>
      <td>0.148816</td>
      <td>0.162204</td>
      <td>0.181771</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.000000</td>
      <td>0.980458</td>
      <td>0.470350</td>
      <td>0.235849</td>
      <td>0.035040</td>
      <td>0.182615</td>
      <td>0.330189</td>
      <td>0.359838</td>
      <td>0.382075</td>
      <td>0.396226</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.949455</td>
      <td>0.764270</td>
      <td>0.242266</td>
      <td>0.086710</td>
      <td>0.087582</td>
      <td>0.138562</td>
      <td>0.161220</td>
      <td>0.155120</td>
      <td>0.145969</td>
      <td>0.148584</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.914286</td>
      <td>0.922286</td>
      <td>0.590857</td>
      <td>0.149714</td>
      <td>0.000000</td>
      <td>0.236000</td>
      <td>0.321714</td>
      <td>0.386857</td>
      <td>0.414286</td>
      <td>0.433714</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4041</th>
      <td>1.000000</td>
      <td>0.886778</td>
      <td>0.326257</td>
      <td>0.071136</td>
      <td>0.093855</td>
      <td>0.162756</td>
      <td>0.152700</td>
      <td>0.147486</td>
      <td>0.142272</td>
      <td>0.141899</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4042</th>
      <td>1.000000</td>
      <td>0.770000</td>
      <td>0.395652</td>
      <td>0.199130</td>
      <td>0.015652</td>
      <td>0.046957</td>
      <td>0.117826</td>
      <td>0.147826</td>
      <td>0.146522</td>
      <td>0.149130</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4043</th>
      <td>0.961121</td>
      <td>0.524864</td>
      <td>0.051537</td>
      <td>0.000452</td>
      <td>0.000000</td>
      <td>0.089964</td>
      <td>0.147378</td>
      <td>0.146022</td>
      <td>0.153255</td>
      <td>0.158228</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4044</th>
      <td>0.975960</td>
      <td>0.867241</td>
      <td>0.465734</td>
      <td>0.212774</td>
      <td>0.000718</td>
      <td>0.001435</td>
      <td>0.067815</td>
      <td>0.083961</td>
      <td>0.081091</td>
      <td>0.083602</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4045</th>
      <td>1.000000</td>
      <td>0.535237</td>
      <td>0.317878</td>
      <td>0.080490</td>
      <td>0.000000</td>
      <td>0.060460</td>
      <td>0.122774</td>
      <td>0.114985</td>
      <td>0.114614</td>
      <td>0.112389</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>4046 rows × 187 columns</p>
</div>




```python
print('\t\tNumber of ECG samples\nHealthy:\t{}\nAbnormal:\t{}'.format(len(dfN), len(dfA)))
```

    		Number of ECG samples
    Healthy:	4046
    Abnormal:	10506
    


```python
def visualize_waveform(waveform, std=None):
    sampling_frequenzy = 125 # Hz
    time = np.arange(len(waveform)) / sampling_frequenzy * 1000. # relative timescale
    
    plt.clf()
    plt.plot(time, waveform, color='k')
    plt.xlabel("Time (ms)")
    plt.ylabel("Normalized ECG value")
    
    if std is not None:
        # Plot standard deviation
        plt.fill_between(
            time,
            waveform-std,
            waveform+std,
            color='k',
            alpha=0.4
        )
        
        
    plt.show()
```


```python
idx = random.choice(range(len(dfN)))
print('Visualize extracted beat example {} (healthy)'.format(idx))

waveform = dfN.iloc[idx, :].to_numpy()
visualize_waveform(waveform)

idx = random.choice(range(len(dfA)))
print('Visualize extracted beat example {} (abnormal)'.format(idx))

waveform = dfA.iloc[idx, :].to_numpy()
visualize_waveform(waveform)
```

    Visualize extracted beat example 2194 (healthy)
    


    
![png](output_8_1.png)
    


    Visualize extracted beat example 926 (abnormal)
    


    
![png](output_8_3.png)
    


The next figures visualize the mean waveforms and their corresponding standard deviations (gray bands) for the healty and abnormal classes, respectively. The two mean waveforms are not very distinct and have large dispersions, which will reduce the efficiency of the anomaly detection. The difference between the waveforms might be more visible when comparing the healthy waveform to the indiviudal diagnostic classes. For this dataset, the dominant class is myocardial infarction. We note that the distinction between the two waveforms might be improved by a different preprocessing of the raw data. 


```python
print('Visualize mean waveform (healthy)')
visualize_waveform(dfN.mean().to_numpy(), dfN.std().to_numpy())
print('Visualize mean waveform (abnormal)')
visualize_waveform(dfA.mean().to_numpy(), dfA.std().to_numpy())
```

    Visualize mean waveform (healthy)
    


    
![png](output_10_1.png)
    


    Visualize mean waveform (abnormal)
    


    
![png](output_10_3.png)
    



```python
# Split the dataset of healthy ECG beats into training, validation and test subsets
train_df, val_df = train_test_split(dfN, test_size=0.2)
val_df, test_df = train_test_split(val_df, test_size=0.4)

ntot = len(dfN)
print('Training: \t{:.2f}% ({})'.format(len(train_df)/ntot*100., len(train_df)))
print('Validation: \t{:.2f}% ({})'.format(len(val_df)/ntot*100., len(val_df)))
print('Test: \t\t{:.2f}% ({})'.format(len(test_df)/ntot*100., len(test_df)))
```

    Training: 	79.98% (3236)
    Validation: 	12.01% (486)
    Test: 		8.01% (324)
    


```python
def create_dataset(df):
    # This approach corresponds effectively to a batch size of 1
    # Convert to tensor
    # The unsqueeze command converts each data point in a tensor
    waveforms = df.astype(np.float32).to_numpy()
    dataset = [torch.tensor(wf).unsqueeze(1) for wf in waveforms] 
    
    n_seq, seq_len, n_features = torch.stack(dataset).shape
    
    return dataset, seq_len, n_features
```


```python
train_dataset, seq_len, n_features = create_dataset(train_df)
val_dataset, _, _ = create_dataset(val_df)

test_N_dataset, _, _ = create_dataset(test_df)
test_A_dataset, _, _ = create_dataset(dfA)
```

### The LSTM autoencoder
In the encoding step, an autoencoder learns to represent the input data by a lower dimensional number of features. From this representation, the decoder step reconstructs the original input as close as possible. This way, the model learns the most important features of the data, meaning a compressed representation.
In this example, the data points of the ECG have a temporal dependency. We model this using two LSTM layers.


```python
class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim):
        super(Encoder, self).__init__()
        
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.hidden_dim = 2 * embedding_dim
        
        # Define two LSTM layers to encode the time series data
        self.rnn1 = nn.LSTM(
            input_size = n_features,
            hidden_size = self.hidden_dim,
            num_layers = 1,
            batch_first = True
        )
        
        self.rnn2 = nn.LSTM(
            input_size = self.hidden_dim,
            hidden_size = embedding_dim,
            num_layers = 1,
            batch_first = True
        )
    
    def forward(self, x):
        x = x.reshape((1, self.seq_len, self.n_features))
        x, (_, _) = self.rnn1(x)
        x, (hidden_n, _) = self.rnn2(x)
        return hidden_n.reshape((self.n_features, self.embedding_dim))
```


```python
class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim, n_features):
        super(Decoder, self).__init__()
        
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.n_features = n_features

        self.hidden_dim = 2 * input_dim
        
        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=1,
            batch_first=True
        )
        
        self.rnn2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        self.output_layer = nn.Linear(self.hidden_dim, n_features)
        
    def forward(self, x):
        x = x.repeat(self.seq_len, self.n_features)
        x = x.reshape((self.n_features, self.seq_len, self.input_dim))
        x, (_, _) = self.rnn1(x)
        x, (_, _) = self.rnn2(x)
        x = x.reshape((self.seq_len, self.hidden_dim))
        return self.output_layer(x)
```


```python
class RecurrentAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim):
        super(RecurrentAutoencoder, self).__init__()
        self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
        self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
```


```python
def load_model():
    # To track loss for each epoch
    loss_history = dict(train=[], val=[])
    
    # Learning rate
    lrn_rate = .01
        
    if os.path.isfile('model.pth'):
        # Load model and loss history
        model = torch.load('model.pth')
        loss_df = pd.read_csv('loss_history.csv', index_col=0)
        loss_history['train'] = loss_df.train.tolist()
        loss_history['val'] = loss_df.val.tolist()
        
        lrn_rate = lrn_rate * np.power(lrn_gamma, len(loss_history['val']))
        val_loss_min = np.min(loss_history['val'])
    else:
        model = RecurrentAutoencoder(seq_len, n_features, 128)
        # To track validation loss changes
        val_loss_min = np.Inf      
    
    model = model.to(device)
    return model, loss_history, val_loss_min, lrn_rate
```


```python
lrn_gamma = 0.99

model, loss_history, val_loss_min, lrn_rate = load_model()
```


```python
model
```




    RecurrentAutoencoder(
      (encoder): Encoder(
        (rnn1): LSTM(1, 256, batch_first=True)
        (rnn2): LSTM(256, 128, batch_first=True)
      )
      (decoder): Decoder(
        (rnn1): LSTM(128, 128, batch_first=True)
        (rnn2): LSTM(128, 256, batch_first=True)
        (output_layer): Linear(in_features=256, out_features=1, bias=True)
      )
    )



### Train the network
As input for the training, we just need the healthy waveforms. The model learns to represent this data in an unsupervised manner by minimizing the reconstruction loss. We use a standard L1 Loss function (Least Absolute Deviations) to minimize the sum of all the absolute differences between the true value and the predicted value.


```python
criterion = nn.L1Loss(reduction="sum").to(device)
optimizer = optim.Adam(model.parameters(), lr=lrn_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=lrn_gamma)
```


```python
# Specify the number of epochs to train the model
n_epochs = 10

# Set start value
n_start = len(loss_history['val'])+1

for n_loop, epoch in enumerate(range(n_start, n_start+n_epochs)):
    # Model training
    model.train()
    
    train_losses = []
    for waveform in train_dataset:
        optimizer.zero_grad()
        waveform = waveform.to(device)
        prediction = model(waveform)
        loss = criterion(prediction, waveform)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    
    # Model validation
    val_losses = []
    model.eval()
    with torch.no_grad():
        for waveform in val_dataset:
            waveform = waveform.to(device)
            prediction = model(waveform)
            loss = criterion(prediction, waveform)
            val_losses.append(loss.item())
    
    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)
    
    loss_history['train'].append(train_loss)
    loss_history['val'].append(val_loss)
    
    # Save the model if validation loss decreases
    if val_loss < val_loss_min:
        torch.save(model, 'model.pth')
        val_loss_min = val_loss
    
    # Adjust learning rate
    curr_lr = scheduler.get_last_lr()
    scheduler.step()
    
    if n_loop == 0:
        print('Epoch \tTraining loss \tValidation loss \tLearning rate\n')
              
    print('{} \t{:.4f} \t{:.4f} \t\t{:.4f}'.format(epoch,train_loss, val_loss, curr_lr[0]))

loss_df = pd.DataFrame.from_dict(loss_history)
loss_df.to_csv('loss_history.csv', index=True)
```

    11 	28.2708 	27.8928 		0.0090
    12 	28.2097 	29.1774 		0.0090
    13 	27.9230 	28.4031 		0.0089
    14 	28.2040 	30.7012 		0.0088
    15 	28.3351 	28.3568 		0.0087
    16 	27.9865 	23.0098 		0.0086
    17 	28.0558 	25.8536 		0.0085
    18 	27.9899 	27.2267 		0.0084
    19 	27.3876 	22.7765 		0.0083
    20 	28.0168 	24.7474 		0.0083
    

### Visualize loss history
In this first run, the training loss decreases with the number of epochs, which is desired. However, the number of epochs needs to be increased for convergence. The validation loss scatters a lot, indicating that the validation set is too small. Here, we are interested in the implementation of the LSTM autoencoder and do not continue training the model to improve the results.


```python
epochs = loss_df.index.tolist()

plt.clf()

plt.plot(epochs, loss_df['train'].to_numpy(), ls='-', label='Training')
plt.plot(epochs, loss_df['val'].to_numpy(), ls='--', label='Validation')

plt.xlabel("Epoch")
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.show()
```


    
![png](output_25_0.png)
    


### Evaluate the training datasets
We choose a threshold of the reconstruction error (the loss) to make the prediction whether a heartbeat is an anomaly or not. First, we visualize the distribution of the losses of the training dataset. The majority of the waveforms are reconstructed with a loss below 30.
Therefore, we classify heartbeats below this threshold as healthy and heartbeats with larger losses as anomalies. This choice is a trade-off between precision and sensitivity towards a given class and needs to be fine-tuned for a real world task. For example, one can be more conservative by lowering the threshold. This would result in more false positives. On the other hand, increasing the threshold will reduce the number of false positives but increase the probability to miss actual anomalies.

We note again, that the model is not converged and thus the reconstruction-loss distribution is broad and a large number of heartbeats have a large reconstruction loss.


```python
def make_pretiction(dataset, model):
    predictions, losses = [], []
    
    model.eval()
    with torch.no_grad():
        for waveform in dataset:
            waveform.to(device)
            prediction = model(waveform)
            loss = criterion(prediction, waveform)
            
            predictions.append(prediction.cpu().numpy().flatten())
            losses.append(loss.item())
    return predictions, losses
```


```python
_, train_losses = make_pretiction(train_dataset, model)
```


```python
ax = sns.histplot(data=train_losses, bins=20, kde=True)
ax.set(xlabel='Reconstruction loss', ylabel='Count', title='Training dataset')
plt.show()
```


    
![png](output_29_0.png)
    



```python
test_N_pred , test_N_losses = make_pretiction(test_N_dataset, model)
# Use a subset of the anomaly set for quick testing
test_A_pred , test_A_losses = make_pretiction(test_A_dataset[:len(test_N_dataset)], model)
```


```python
threshold = 30.

test_losses = np.concatenate((test_N_losses, test_A_losses))
prediction = [0 if loss <= threshold else 1 for loss in test_losses]

label = np.concatenate((np.zeros(len(test_N_losses)), np.ones(len(test_A_losses))))
```

### Accuracy, Precision, Recall & F1-Score
In our example, a label of 1 (positive) represents the anomalies.

#### True Negatives:
Number of correctly predicted negatives out of all negatives in the test dataset

#### False Positives:
Number of incorrect positive predictions

#### False Negatives:
Number of incorrect negative predictions

#### True Positives:
Number of correctly predicted positives out of all positves in the test dataset


```python
conf_matrix = confusion_matrix(y_true=label, y_pred=prediction)
matrix_label = ['True Negatives', 'False Positives', 'False Negatives', 'True Positives']

for ml, cm in zip(matrix_label, conf_matrix.ravel()):
    print('{}: \t{}'.format(ml,cm))
```

    True Negatives: 	277
    False Positives: 	47
    False Negatives: 	191
    True Positives: 	133
    

#### Precision score:
Represents the model’s ability to correctly predict positives:
Precision = TP / (FP + TP)

#### Recall score:
Represents the model’s ability to correctly predict positives out of actual positives:
Recall = TP / (FN + TP)

#### Accuracy score:
Represents the model’s ability to correctly predict positives and negatives out of all predictions:
Accuracy = (TP + TN)/ (TP + FN + TN + FP)

#### F1 score:
Represents a weighted average of precision and recall:
F1 = 2 x Precision x Recall / (Precision + Recall)


```python
print('Precision: \t{:.3f}'.format(precision_score(label, prediction)))
print('Recall: \t{:.3f}'.format(recall_score(label, prediction)))
print('Accuracy: \t{:.3f}'.format(accuracy_score(label, prediction)))
print('F1: \t\t{:.3f}'.format(f1_score(label, prediction)))
```

    Precision: 	0.739
    Recall: 	0.410
    Accuracy: 	0.633
    F1: 		0.528
    


```python

```
