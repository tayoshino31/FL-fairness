import numpy as np

def create_combined_data(global_data):
    combined_data = {
        'A0': {'x': [], 'y': [], 's': []},
        'A1': {'x': [], 'y': [], 's': []},
        'B0': {'x': [], 'y': [], 's': []},
        'B1': {'x': [], 'y': [], 's': []}}
    for data in global_data:
        if data is not None:
            for group in ['A0', 'A1', 'B0', 'B1']:
                combined_data[group]['x'].extend(data[group]['x'])
                combined_data[group]['y'].extend(data[group]['y'])
                combined_data[group]['s'].extend(data[group]['s'])
    return combined_data

def convert_xys(data):
    x, y, s = [], [], []
    for group in ['A0', 'A1', 'B0', 'B1']:
        x.extend(data[group]['x'])
        y.extend(data[group]['y'])
        s.extend(data[group]['s'])  
    return np.array(x), np.array(y), np.array(s)
