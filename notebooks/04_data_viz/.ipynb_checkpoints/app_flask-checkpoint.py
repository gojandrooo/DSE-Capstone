import sys
sys.path.append("../../model/")
from flask import Flask,request
import json
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np
import ModelNN as Net
import torch
app = Flask(__name__)

#intialize values
seed=27
data_dir = '../../data/'
model_dir = '../../model/'
# nn_model = 'two_layer'
nn_model = 'TwoLayer_750_epochs_optimized_roc_auc_score'
peak_mmh = 'peak_i15_mmh'

# data_file = data_dir + 'data_v09_consolidated.parquet'
data_file = data_dir + 'data_v08_consolidated.parquet'

#read parameters from config file
with open(model_dir + "model_parameters.json","r") as jsonfile:
    params= json.load(jsonfile)[nn_model]

#read model data from file 
X_train_df = pd.read_parquet(data_file)
y_train_df = X_train_df['response']

#select only required columns/features
nn_data = X_train_df[params['features']]    
col_indx = nn_data.columns.get_loc(peak_mmh)

#scale data 
ss = StandardScaler()
nn_data = ss.fit_transform(nn_data)

#now get the mean and std dev for rain internsity parameter ( from already fitted StandardScaler class)
i15_feature_indx = ss.feature_names_in_.tolist().index(peak_mmh)
i15_mean= ss.mean_.tolist()[i15_feature_indx]
i15_std = np.sqrt(ss.var_.tolist()[i15_feature_indx])

#intialize model params
input_size = nn_data.shape[1]
hidden_size = params['hidden_size']
learning_rate = params['lr'] 
dropout_rate = params['dropout_rate']
output_size = 1 

#intialize model architechure
model = Net.TwoLayer(input_size, hidden_size, output_size, dropout_rate)

#load weights

# model.load_state_dict(torch.load(model_dir+params["weights"]))
if torch.cuda.is_available():
    model.load_state_dict(torch.load(model_dir+params["weights"]))
else:
    model.load_state_dict(torch.load(model_dir+params["weights"],map_location=torch.device('cpu')))

model.eval()

#load the test data 
#this file will have one row for each site ( 693 records)

site_file = data_dir + 'sites_v02_plot_data.parquet'

X_test_df = pd.read_parquet(site_file)
nn_data = X_test_df[params['features']]  

#scale the test data with same scaler as used to 
nn_data = ss.transform(nn_data)

#URL Binding     
@app.route('/predictForAllSites/', methods=['GET'])
def predictForAllSites():
    
    # read parameters from the URL
    parameters = request.args.to_dict()
    i15 = float(parameters['rainFall'])
    
    #scale the rainFall parameter
    peak_i15_scaled = (i15 - i15_mean)/i15_std
    
    #update the data with peak_i15_scaled value for all sites
    #we are predicting the DF across all sites for given amount of storm rainfall
    nn_data[:,col_indx] = peak_i15_scaled
    
    tensor_data = torch.from_numpy(nn_data).float()

    #use model to predict
    y_pred = model(tensor_data)
    
    #convert y_pred tensor to list 
    df_prob_list =  [np.round(i[0],2) for i in y_pred.detach().numpy().tolist()]
    
    #return the list of probabilities
    return df_prob_list

app.run()