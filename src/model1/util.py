import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import torch

# global vars
country_names = []


def load_dataset():
    print("Loading Dataset... ", end="")
    DatasetInput, DatasetLabel, DatasetState, DatasetAux = parse_dataset()
    print("Completed!")
    
    return DatasetInput, DatasetLabel, DatasetState, DatasetAux


def parse_dataset():
    global country_names
    
    # return training/testing data list
    data_list = []
    label_list = []
    state_list = []
    aux_list = []
    
    # load npz file
    covid_data = np.load('data/covid_data.npz', allow_pickle=True)
    
    # load data files in npz
    country_names = covid_data['loc_name']
    data = covid_data['data']
    
    # construct map: country_name -> target_idx
    country_idx_map = {}
    for i, name in enumerate(country_names):
        country_idx_map[name] = i
    #print(country_idx_map)
    
    # TODO make this as parameter of this function
    target_countries = ['South Korea', 'United States', 'Germany']
    #target_countries = [k for k, _ in country_idx_map.items()]
    
    
    
    # read contents in the data files for each country
    for name in target_countries: # FIX for debug
        idx = country_idx_map[name]
        
        country_data = data[idx]
        
        # daily status
        country_date = country_data['dates']
        country_positive = country_data['new_case']
        country_death = country_data['new_death']
        country_icu = country_data['icu']
        country_hosp = country_data['hosp_patients']
        
        # accumulated data
        country_positive_acc = np.add.accumulate(country_positive)
        country_death_acc = np.add.accumulate(country_death)
        
        # auxiliary data (scalar data)
        population = country_data['population']
        med_age = country_data['median_age']
        age65_older = country_data['aged_65_older']
        gdp_per_capita = country_data['gdp_per_capita']
        beds = country_data['hospital_beds_per_thousand']
        
        # find length of data field
        data_len = country_date.shape[0]
        ones = np.ones(data_len)
        
        
        # input data
        stacked_data = np.stack([#country_date,
                                 country_positive,
                                 country_death,
                                 country_icu,
                                 country_hosp,
                                 population * ones,
                                 med_age * ones,
                                 age65_older * ones,
                                 gdp_per_capita * ones,
                                 beds * ones], axis=1)
        #print(stacked.shape)
        
        # label data
        label = np.stack([country_positive_acc,
                          country_death_acc], axis=1)
        
        # data auxiliary informations
        #print(country_date.shape)
        country_date = process_date(country_date)
        #print(country_date.dtype)
        #print(country_date.shape)
        ones = np.ones(data_len, dtype=np.int32)
        aux = np.stack([idx * ones,
                        country_date], axis=1)
        
        # set window size
        WINDOW_SIZE = 30
        for start_idx in range(0, data_len - WINDOW_SIZE * 2 - 1):
            #print(stacked_data.shape)
            #print(label.shape)
            data_for_ml = stacked_data[start_idx:start_idx + WINDOW_SIZE, :]
            label_for_ml = label[start_idx:start_idx + WINDOW_SIZE * 2, :]
            aux_for_vis = aux[start_idx:start_idx + WINDOW_SIZE * 2, :]
            
            #print(data_for_ml.shape)
            
            # Examine initial states (for odeint)    
            # S, E, I, N, P, Hm, Hs, R, D    
            E = np.sum(country_positive[ max(start_idx - 2, 0) : start_idx]) * 2 # exposed, no symptoms (assume 2 days)
            I = np.sum(country_positive[ max(start_idx - 3, 0) : max(start_idx - 2, 0) ]) * 2 # infectious, no symptoms (assume 1 day)
            N = np.sum(country_positive[ max(start_idx - 14, 0) : max(start_idx - 3, 0) ]) # not catched cases
            P = np.sum(country_positive[ max(start_idx - 4, 0) : max(start_idx - 3, 0) ]) # positive cases
            Hm = np.sum(country_positive[ max(start_idx - 14, 0) : max(start_idx - 4, 0) ]) * 0.75 # hospitalise with moderate symptoms
            Hs = np.sum(country_positive[ max(start_idx - 14, 0) : max(start_idx - 4, 0) ]) * 0.25 # hospitalise with severe sympotms
            D = np.sum(country_death[:start_idx])
            R = np.sum(country_positive[max(start_idx - 180, 0) : max(start_idx - 14, 0) ]) - D
            
            #Tot_positive = np.sum(country_positive[:start_idx + 1])
            #Tot_death = np.sum(country_death[:start_idx + 1])
            
            S = population - (E + I + N + P + Hm + Hs + D + R)
            
            x0 = np.array([S, E, I, N, P, Hm, Hs, R, D]) # initial states
            
            #print(name, x0)
            #print(label)
            #print(label_for_ml)
            
            #print(data_for_ml.shape)
            #print(label_for_ml.shape)
            #print(x0.shape)
            
            data_list.append(data_for_ml)
            label_list.append(label_for_ml)
            state_list.append(x0)
            aux_list.append(aux_for_vis)
            
    
    data_list = np.array(data_list)
    label_list = np.array(label_list)
    state_list = np.array(state_list)
    aux_list = np.array(aux_list)
    
    #print(data_list.shape)
    #print(label_list.shape)
    #print(state_list.shape)
    
    #print("Aux!", aux_list.shape)
    
    return data_list, label_list, state_list, aux_list


def unpack_dataset(data_input, data_label, data_state, data_aux, val_idx=0):
    
    N, _, = data_state.shape
    
    
    valid_input = data_input[val_idx]
    valid_label = data_label[val_idx]
    valid_state = data_state[val_idx]
    valid_aux = data_aux[val_idx]
    
    train_input = data_input#np.delete(data_input, val_idx)
    train_label = data_label#np.delete(data_label, val_idx)
    train_state = data_state#np.delete(data_state, val_idx)
    train_aux = data_aux
    
    # TODO make test dataset!
    
    return train_input, train_label, train_state, train_aux, valid_input, valid_label, valid_state, valid_aux
    

from enum import Enum
class SEIRD_Param(Enum):
    alpha_E = 0
    alpha_I = 1
    rho = 2
    kappa = 3
    beta_I = 4
    beta_N = 5
    gamma_N = 6
    gamma_M = 7
    gamma_S = 8
    delta = 9
    mu = 10 
class SEIRD_State(Enum):
    S = 0
    E = 1
    I = 2
    N = 3
    P = 4
    Hm = 5
    Hs = 6
    R = 7
    D = 8


def odeint_plot(output, batch_idx=0, vis=True, save=False, epoch=0, aux=1):
    ax = plt.subplot()
    
    batch, num_t, num_state = output.shape
    
    output = output[batch_idx,:,:]
    output = output.detach().numpy()
    
    start_date = aux[batch_idx, 0, 1]
    country_name = country_names[aux[batch_idx, 0, 0]]
    
    # plot states
    for i in range(9):
        ax.plot(output[:,i], label=str(SEIRD_State(i))[12:])
    
    
    ax.legend()
    ax.set_title("epoch %3d - (%s, %d)" % (int(epoch), country_name, start_date), fontsize=15)
    
    if save:
        plt.savefig('out/epoch_%03d.png' % int(epoch))
        print("[odeint_plot] RESULT SAVED")
    if vis:
        plt.show()


def check_device():
    print("### Device Check list ###")
    print("GPU available?:", torch.cuda.is_available())
    if torch.cuda.is_available():
        device_number = torch.cuda.current_device()
        print("Device number:", device_number)
        print("Is device?:", torch.cuda.device(device_number))
        print("Device count?:", torch.cuda.device_count())
        print("Device name?:", torch.cuda.get_device_name(device_number))
        print("### ### ### ### ### ###\n\n")


def process_date(dates):
    rtn = np.array([0], dtype=np.int32)
    
    for date in dates:
        temp = "".join(date.split('-'))
        num_date = int(temp)
        #print(num_date)
        rtn = np.concatenate((rtn, [num_date]), dtype=np.int32)

    return rtn[1:]