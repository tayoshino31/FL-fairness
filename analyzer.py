import numpy as np
import matplotlib.pyplot as plt
import yaml
from ipywidgets import interactive_output, FloatSlider, HBox, VBox, Label, Layout, Output
from IPython.display import display, clear_output
from trainers.trainer import standalone, centralized, fedavg, bruteforce, eval_local
from utils.data_generator import Data_Generator
from utils.heplers import create_combined_data, convert_xys

class Analyzer:
    def __init__(self, n_client, fedavg_rounds, fedavg_local_epochs, fedavg_local_lr, centralized_epochs, centralized_lr, bf_range, bf_step):
        self.n_client = n_client
        self.centralized_lr = centralized_lr
        self.fedavg_rounds =  fedavg_rounds
        self.fedavg_local_epochs = fedavg_local_epochs 
        self.fedavg_local_lr = fedavg_local_lr
        self.prev_params = [None, None]
        self.prev_data = [None, None]
        
        self.centralized_epochs = centralized_epochs
        self.bf_range = bf_range
        self.bf_step = bf_step
        with open('config.yaml', 'r') as config_file:
            config = yaml.safe_load(config_file)
        self.sliders_config = config["sliders_config"]
        self.layout_groups = config["layout_groups"]
        self.global_data = [None] * n_client    
        self.client_outputs = [Output() for _ in range(self.n_client)]
        self.client_layouts = []
        self.global_layout = Output()
        self.bf_results = [None,None]
         
        # initialize client and global model.
        self.init_clients()
        self.update_global()
        
        
    ##### UI setting ###########################################################
    def create_client_ui(self, client_idx):
        # create dicts of sliders  
        # eg. 'N': FloatSlider(min=100, max=20000, step=100, value=10000, description='N'),
        sliders = {key: FloatSlider(**self.sliders_config[key]) for key in self.sliders_config}
        # set the sliders' placements.
        slider_placements = [HBox([sliders[key] for key in group]) for group in self.layout_groups]
        slider_titles = Label(f'Client {client_idx} Parameters', layout=Layout(height='30px', 
                                align_self='center', justify_content='center'))
        slider_ui = VBox([slider_titles] + slider_placements)
        # set interactive graph and slider ui.
        output_graph = interactive_output(lambda **kwargs: self.update_client(client_idx, kwargs), sliders)
        client_layout = VBox([output_graph, HBox([self.client_outputs[client_idx], slider_ui])])
        return client_layout
        
    def init_clients(self):
        for client_idx in range(self.n_client):
            client_layout = self.create_client_ui(client_idx)
            self.client_layouts.append(client_layout)
        
    def draw_client(self, sa_result, bf_result, client_data, client_idx):
        clear_output(wait=True) 
        fig, ax = plt.subplots(figsize=(6, 2.5))
        ax.set_title(f'Client {client_idx}')
        textstr = '\n'.join((
            f'$r:standalone={sa_result[3]:.3f}$', 
            f'$Acc={sa_result[0]:.4f}$',
            f'$EO={sa_result[1]:.3f}$', 
            f'$DP={sa_result[2]:.3f}$', 
            f'$g:bruteforce={bf_result[3]:.3f}$', 
            f'$Acc={bf_result[0]:.4f}$',
            f'$EO={bf_result[1]:.3f}$', 
            f'$DP={bf_result[2]:.3f}$'))
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
        ax.set_xlim(-10, 10)
        ax.set_ylim(0, 1)
        ax.axvline(x=sa_result[3], color='r', linestyle='--') 
        ax.axvline(x=bf_result[3], color='g', linestyle='--') 
        ax.hist(client_data['A0']['x'], bins=100, alpha=0.5, density=True, label='Group A, Y = 0', color='red')
        ax.hist(client_data['B0']['x'], bins=100, alpha=0.5, density=True, label='Group B, Y = 0', color='orange')
        ax.hist(client_data['A1']['x'], bins=100, alpha=0.5, density=True, label='Group A, Y = 1', color='blue')
        ax.hist(client_data['B1']['x'], bins=100, alpha=0.5, density=True, label='Group B, Y = 1', color='green')
        ax.legend(loc='upper right')
        ax.set_ylabel('Density')
        plt.show()
        
    def draw_global(self, centralized_result, fedavg_result, bf_result, data, 
                    standalone_local, centralized_local, fedavg_local, bf_local):
        clear_output(wait=True) 
        fig, ax = plt.subplots(figsize=(16, 6))
        plt.title('Global Data')
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        
        #centralized, fedavg, bruteforce
        textstr= '\n'.join([
            "global",
            f'$r:centralized={centralized_result[3]:.3f}$', 
            f'$Acc={centralized_result[0]:.4f}$', 
            f'$EO={centralized_result[1]:.3f}$', 
            f'$DP={centralized_result[2]:.3f}$',
            f'$b:fedavg={fedavg_result[3]:.3f}$', 
            f'$Acc={fedavg_result[0]:.4f}$', 
            f'$EO={fedavg_result[1]:.3f}$', 
            f'$DP={fedavg_result[2]:.3f}$',
            f'$g:bruteforce={bf_result[3]:.3f}$', 
            f'$Acc={bf_result[0]:.4f}$',
            f'$EO={bf_result[1]:.3f}$',
            f'$DP={bf_result[2]:.3f}$',
        ])
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
        
        #local data evaluation
        standalone_clinet0, standalone_client1 = standalone_local[0], standalone_local[1]
        centralized_client0, centralized_client1 = centralized_local[0], centralized_local[1]
        fedavg_client0, fedavg_client1 = fedavg_local[0], fedavg_local[1]
        bf_client0, bf_client1 = bf_local[0], bf_local[1]
        
        textstr_local0 = '\n'.join([
            "client_0",
            f'$standalone={standalone_clinet0[3]:.3f}$', 
            f'$Acc={standalone_clinet0[0]:.4f}$', 
            f'$EO={standalone_clinet0[1]:.3f}$', 
            f'$DP={standalone_clinet0[2]:.3f}$',
            f'$centralized={centralized_client0[3]:.3f}$', 
            f'$Acc={centralized_client0[0]:.4f}$', 
            f'$EO={centralized_client0[1]:.3f}$', 
            f'$DP={centralized_client0[2]:.3f}$',
            f'$fedavg={fedavg_client0[3]:.3f}$', 
            f'$Acc={fedavg_client0[0]:.4f}$', 
            f'$EO={fedavg_client0[1]:.3f}$', 
            f'$DP={fedavg_client0[2]:.3f}$',
            f'$bruteforce={bf_client0[3]:.3f}$', 
            f'$Acc={bf_client0[0]:.4f}$', 
            f'$EO={bf_client0[1]:.3f}$', 
            f'$DP={bf_client0[2]:.3f}$',
        ])
        ax.text(0.20, 0.95, textstr_local0, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
        
        #client_1
        textstr_local1 = '\n'.join([
            "client_1",
            f'$standalone={standalone_client1[3]:.3f}$', 
            f'$Acc={standalone_client1[0]:.4f}$',
            f'$EO={standalone_client1[1]:.3f}$',
            f'$DP={standalone_client1[2]:.3f}$',
            f'$centralized={centralized_client1[3]:.3f}$', 
            f'$Acc={centralized_client1[0]:.4f}$',
            f'$EO={centralized_client1[1]:.3f}$',
            f'$DP={centralized_client1[2]:.3f}$',
            f'$fedavg={fedavg_client1[3]:.3f}$', 
            f'$Acc={fedavg_client1[0]:.4f}$',
            f'$EO={fedavg_client1[1]:.3f}$',
            f'$DP={fedavg_client1[2]:.3f}$',
            f'$bruteforce={bf_client1[3]:.3f}$', 
            f'$Acc={bf_client1[0]:.4f}$', 
            f'$EO={bf_client1[1]:.3f}$', 
            f'$DP={bf_client1[2]:.3f}$',
        ])
        ax.text(0.35, 0.95, textstr_local1, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
        
        ax.set_xlim(-10, 10)
        ax.set_ylim(0, 1)
        ax.axvline(x=fedavg_result[3],  color='b', linestyle='--') 
        ax.axvline(x=centralized_result[3], color='r', linestyle='--') 
        ax.axvline(x=bf_result[3], color='g', linestyle='--') 
        ax.hist(data['A0']['x'], bins=100, alpha=0.5, density=True, label='Group A, Y = 0', color='red')
        ax.hist(data['B0']['x'], bins=100, alpha=0.5, density=True, label='Group B, Y = 0', color='orange')
        ax.hist(data['A1']['x'], bins=100, alpha=0.5, density=True, label='Group A, Y = 1', color='blue')
        ax.hist(data['B1']['x'], bins=100, alpha=0.5, density=True, label='Group B, Y = 1', color='green')
        ax.legend(loc='upper right')
        ax.set_ylabel('Density')
        plt.show()
    
    
    #### Model training & evaluation ###########################################
    def is_updated(self, curr_params, prev_params):
        param_names = ['N', 'n_a', 'alpha_a', 'alpha_b',
                       'mean_A1', 'mean_A0', 'mean_B1', 'mean_B0', 
                       'std_A1', 'std_A0', 'std_B1','std_B0']
        for param_name in param_names:
            if(curr_params[param_name] != prev_params[param_name]):
                return True
        return False
    
    def update_client(self, client_idx, params):
        # generate training data based on the given parameters.
        means = [params['mean_A1'], params['mean_A0'], params['mean_B1'],params['mean_B0']]
        stds = [params['std_A1'], params['std_A0'], params['std_B1'],params['std_B0']]
        lr = params['st_lr']
        self.fedavg_local_lr[client_idx] = params['fed_lr']
        self.fedavg_local_epochs[client_idx] = params['fed_epochs']
        
        # generate clients data
        if(self.prev_data[client_idx] is None or self.is_updated(params, self.prev_params[client_idx])):
            data = Data_Generator(params['N'], params['n_a'], params['alpha_a'], params['alpha_b'], means, stds)
        else:
            data = self.prev_data[client_idx]
        client_data = data.get_client()
        x, y, s = data.get_xys()
        self.prev_params[client_idx] = params
        self.prev_data[client_idx] = data
        
        # train and evaluate standalone and bruteforce models.
        sa_result = standalone(x, y, s, lr, epochs=int(params['st_epochs']))
        bf_result = bruteforce(x, y, s, self.bf_range, self.bf_step, warm_start=sa_result[3])
        self.bf_results[client_idx] = bf_result
        
        # update global data.
        self.global_data[client_idx] = client_data
        # update specified clident model.
        with self.client_outputs[client_idx]:
            self.draw_client(sa_result, bf_result, client_data, client_idx)
        # update global model corresponding to the change of global data.
        self.update_global()
    
    def update_global(self):
        # adjust data format
        combined_data = create_combined_data(self.global_data)
        x, y, s = convert_xys(combined_data)
        # train and eval centralized, fedavg, and bruteforce models.
        centralized_result = centralized(combined_data, self.centralized_lr, self.centralized_epochs)
        fedavg_result = fedavg(combined_data, self.global_data, self.fedavg_rounds, self.fedavg_local_lr, self.fedavg_local_epochs)
        bf_result = bruteforce(x, y, s, self.bf_range, self.bf_step, warm_start=centralized_result[3])
        #evaluate fedavg and centralized on local data.
        client0_data = convert_xys(self.global_data[0])
        client1_data = convert_xys(self.global_data[1])
        
        client0_centralized = eval_local(client0_data, centralized_result[3])
        client1_centralized = eval_local(client1_data, centralized_result[3])
        client0_standalone = eval_local(client0_data, self.bf_results[0][3])
        client1_standalone = eval_local(client1_data, self.bf_results[1][3])
        client0_fedavg = eval_local(client0_data,fedavg_result[3])
        client1_fedavg = eval_local(client1_data,fedavg_result[3])
        client0_bf_local = eval_local(client0_data,bf_result[3])
        client1_bf_local = eval_local(client1_data,bf_result[3])
        
        # update the result.
        with self.global_layout:
            self.draw_global(centralized_result, fedavg_result, bf_result, combined_data, 
                             [client0_standalone, client1_standalone],
                             [client0_centralized, client1_centralized],
                             [client0_fedavg, client1_fedavg],
                             [client0_bf_local, client1_bf_local])
    
    #### Display function ######################################################
    def display_result(self):
        layouts = []
        # append all client layouts.
        for client_idx in range(self.n_client):
            layouts.append(self.client_layouts[client_idx])
        # append global layout.
        layouts.append(self.global_layout)
        # display the layouts vertically. 
        display(VBox(layouts))



class BFAnalyzer:
    def __init__(self, n_client, fedavg_rounds, fedavg_local_epochs, fedavg_local_lr, centralized_epochs, centralized_lr, bf_range, bf_step):
        self.n_client = n_client
        self.centralized_lr = centralized_lr
        self.fedavg_rounds =  fedavg_rounds
        self.fedavg_local_epochs = fedavg_local_epochs 
        self.fedavg_local_lr = fedavg_local_lr
        self.prev_params = [None, None]
        self.prev_data = [None, None]
        
        self.centralized_epochs = centralized_epochs
        self.bf_range = bf_range
        self.bf_step = bf_step
        with open('config.yaml', 'r') as config_file:
            config = yaml.safe_load(config_file)
        self.sliders_config = config["sliders_config"]
        self.layout_groups = config["layout_groups"]
        self.global_data = [None] * n_client    
        self.client_outputs = [Output() for _ in range(self.n_client)]
        self.client_layouts = []
        self.global_layout = Output()
        self.bf_results = [None, None]
         
        # initialize client and global model.
        self.init_clients()
        self.update_global()
        
        
    ##### UI setting ###########################################################
    def create_client_ui(self, client_idx):
        # create dicts of sliders  
        # eg. 'N': FloatSlider(min=100, max=20000, step=100, value=10000, description='N'),
        sliders = {key: FloatSlider(**self.sliders_config[key]) for key in self.sliders_config}
        # set the sliders' placements.
        slider_placements = [HBox([sliders[key] for key in group]) for group in self.layout_groups]
        slider_titles = Label(f'Client {client_idx} Parameters', layout=Layout(height='30px', 
                                align_self='center', justify_content='center'))
        slider_ui = VBox([slider_titles] + slider_placements)
        # set interactive graph and slider ui.
        output_graph = interactive_output(lambda **kwargs: self.update_client(client_idx, kwargs), sliders)
        client_layout = VBox([output_graph, HBox([self.client_outputs[client_idx], slider_ui])])
        return client_layout
        
    def init_clients(self):
        for client_idx in range(self.n_client):
            client_layout = self.create_client_ui(client_idx)
            self.client_layouts.append(client_layout)
        
    def draw_client(self, sa_result, bf_result, client_data, client_idx):
        clear_output(wait=True) 
        fig, ax = plt.subplots(figsize=(6, 2.5))
        ax.set_title(f'Client {client_idx}')
        textstr = '\n'.join((
            f'$g:bruteforce={bf_result[3]:.3f}$', 
            f'$Acc={bf_result[0]:.4f}$',
            f'$EO={bf_result[1]:.3f}$', 
            f'$DP={bf_result[2]:.3f}$'))
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
        ax.set_xlim(-10, 10)
        ax.set_ylim(0, 1)
        ax.axvline(x=bf_result[3], color='g', linestyle='--') 
        ax.hist(client_data['A0']['x'], bins=100, alpha=0.5, density=True, label='Group A, Y = 0', color='red')
        ax.hist(client_data['B0']['x'], bins=100, alpha=0.5, density=True, label='Group B, Y = 0', color='orange')
        ax.hist(client_data['A1']['x'], bins=100, alpha=0.5, density=True, label='Group A, Y = 1', color='blue')
        ax.hist(client_data['B1']['x'], bins=100, alpha=0.5, density=True, label='Group B, Y = 1', color='green')
        ax.legend(loc='upper right')
        ax.set_ylabel('Density')
        plt.show()
        
    def draw_global(self, gbf_result, data, bf_local, gbf_local):
        clear_output(wait=True) 
        fig, ax = plt.subplots(figsize=(16, 4))
        plt.title('Global Data')
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        
        #centralized, fedavg, bruteforce
        textstr= '\n'.join([
            "global",
            f'$g:bruteforce={gbf_result[3]:.3f}$', 
            f'$Acc={gbf_result[0]:.4f}$',
            f'$EO={gbf_result[1]:.3f}$',
            f'$DP={gbf_result[2]:.3f}$',
        ])
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
        
        #local data evaluation
        gbf_client0, gbf_client1 = gbf_local[0], gbf_local[1]
        bf_client0, bf_client1 = bf_local[0], bf_local[1]
        
        textstr_local0 = '\n'.join([
            "client_0",
            f'$lbruteforce={bf_client0[3]:.3f}$', 
            f'$Acc={bf_client0[0]:.4f}$', 
            f'$EO={bf_client0[1]:.3f}$', 
            f'$DP={bf_client0[2]:.3f}$',
            f'$gbruteforce={gbf_client0[3]:.3f}$', 
            f'$Acc={gbf_client0[0]:.4f}$', 
            f'$EO={gbf_client0[1]:.3f}$', 
            f'$DP={gbf_client0[2]:.3f}$',
        ])
        ax.text(0.20, 0.95, textstr_local0, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
        
        #client_1
        textstr_local1 = '\n'.join([
            "client_1",
            f'$lbruteforce={bf_client1[3]:.3f}$', 
            f'$Acc={bf_client1[0]:.4f}$', 
            f'$EO={bf_client1[1]:.3f}$', 
            f'$DP={bf_client1[2]:.3f}$',
            f'$gbruteforce={gbf_client1[3]:.3f}$', 
            f'$Acc={gbf_client1[0]:.4f}$', 
            f'$EO={gbf_client1[1]:.3f}$', 
            f'$DP={gbf_client1[2]:.3f}$',
        ])
        ax.text(0.35, 0.95, textstr_local1, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
        
        ax.set_xlim(-10, 10)
        ax.set_ylim(0, 1)
        ax.axvline(x=gbf_result[3], color='g', linestyle='--') 
        ax.axvline(x=bf_client0[3], color='r', linestyle='--') 
        ax.axvline(x=bf_client1[3], color='r', linestyle='--') 
        ax.hist(data['A0']['x'], bins=100, alpha=0.5, density=True, label='Group A, Y = 0', color='red')
        ax.hist(data['B0']['x'], bins=100, alpha=0.5, density=True, label='Group B, Y = 0', color='orange')
        ax.hist(data['A1']['x'], bins=100, alpha=0.5, density=True, label='Group A, Y = 1', color='blue')
        ax.hist(data['B1']['x'], bins=100, alpha=0.5, density=True, label='Group B, Y = 1', color='green')
        ax.legend(loc='upper right')
        ax.set_ylabel('Density')
        plt.show()
    
    
    #### Model training & evaluation ###########################################
    def is_updated(self, curr_params, prev_params):
        param_names = ['N', 'n_a', 'alpha_a', 'alpha_b',
                       'mean_A1', 'mean_A0', 'mean_B1', 'mean_B0', 
                       'std_A1', 'std_A0', 'std_B1','std_B0']
        for param_name in param_names:
            if(curr_params[param_name] != prev_params[param_name]):
                return True
        return False
    
    def update_client(self, client_idx, params):
        # generate training data based on the given parameters.
        means = [params['mean_A1'], params['mean_A0'], params['mean_B1'],params['mean_B0']]
        stds = [params['std_A1'], params['std_A0'], params['std_B1'],params['std_B0']]
        lr = params['st_lr']
        self.fedavg_local_lr[client_idx] = params['fed_lr']
        self.fedavg_local_epochs[client_idx] = params['fed_epochs']
        
        # generate clients data
        if(self.prev_data[client_idx] is None or self.is_updated(params, self.prev_params[client_idx])):
            data = Data_Generator(params['N'], params['n_a'], params['alpha_a'], params['alpha_b'], means, stds)
        else:
            data = self.prev_data[client_idx]
        client_data = data.get_client()
        x, y, s = data.get_xys()
        self.prev_params[client_idx] = params
        self.prev_data[client_idx] = data
        
        # train and evaluate standalone and bruteforce models.
        sa_result = standalone(x, y, s, lr, epochs=int(params['st_epochs']))
        bf_result = bruteforce(x, y, s, self.bf_range, self.bf_step, warm_start=sa_result[3])
        self.bf_results[client_idx] = bf_result
        
        # update global data.
        self.global_data[client_idx] = client_data
        # update specified clident model.
        with self.client_outputs[client_idx]:
            self.draw_client(sa_result, bf_result, client_data, client_idx)
        # update global model corresponding to the change of global data.
        self.update_global()
    
    def update_global(self):
        # adjust data format
        combined_data = create_combined_data(self.global_data)
        x, y, s = convert_xys(combined_data)
        # train and eval centralized, fedavg, and bruteforce models.
        centralized_result = centralized(combined_data, self.centralized_lr, self.centralized_epochs)
        gbf_result = bruteforce(x, y, s, self.bf_range, self.bf_step, warm_start=centralized_result[3])
        #evaluate fedavg and centralized on local data.
        client0_data = convert_xys(self.global_data[0])
        client1_data = convert_xys(self.global_data[1])
        client0_gbf_local = eval_local(client0_data,gbf_result[3])
        client1_gbf_local = eval_local(client1_data,gbf_result[3])
        
        # update the result.
        with self.global_layout:
            self.draw_global(gbf_result, combined_data, 
                             self.bf_results,
                             [client0_gbf_local, client1_gbf_local])
    
    #### Display function ######################################################
    def display_result(self):
        layouts = []
        # append all client layouts.
        for client_idx in range(self.n_client):
            layouts.append(self.client_layouts[client_idx])
        # append global layout.
        layouts.append(self.global_layout)
        # display the layouts vertically. 
        display(VBox(layouts))