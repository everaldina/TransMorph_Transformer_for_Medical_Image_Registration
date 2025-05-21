import pickle
import json

def write2csv(line, name):
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')
        
def save_pickle(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)
        
def json_load(fname):  
    with open(fname) as json_file:
        json_data = json.load(json_file)
        return json_data

def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)