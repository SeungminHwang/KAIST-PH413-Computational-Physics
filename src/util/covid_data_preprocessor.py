import csv
import os
import datetime as dt
import numpy as np


def cond_str_to_float(str):
    if len(str) == 0:
        return 0
    
    return float(str)
def cond_str_to_int(str):
    return int(cond_str_to_float(str))


data_path = "data/owid-covid-data.csv"

f = open(data_path, 'r')
labels = f.readline().split(',')


#print(labels[19])
#assert(False)

reader = csv.reader(f)

data_dict = {}
        
cnt = 0
for line in reader:
    
    # default information
    location = line[2]
    date = line[3]
    new_case = line[5]
    new_death = line[8]
    icu = line[17] # intensive care unit
    hosp_patients = line[19]
    
    
    # auxiliary data
    population = line[48]
    population_density = line[49]
    median_age = line[50]
    aged_65_older = line[51]
    gdp_per_capita = line[53]
    hospital_beds_per_thousand = line[60]
    
    # data conversion (string -> int/float)
    new_case = cond_str_to_int(new_case)
    new_death = cond_str_to_int(new_death)
    icu = cond_str_to_int(icu)
    hosp_patients = cond_str_to_int(hosp_patients)
    
    population = cond_str_to_int(population)
    population_density = cond_str_to_float(population_density)
    median_age = cond_str_to_float(median_age)
    aged_65_older = cond_str_to_float(aged_65_older)
    gdp_per_capita = cond_str_to_float(gdp_per_capita)
    hospital_beds_per_thousand = cond_str_to_float(hospital_beds_per_thousand)
    
    
    # data save
    if location in data_dict:
        # update new daily data
        data_dict[location]['dates'] = np.concatenate ((data_dict[location]['dates'], [date]))
        data_dict[location]['new_case'] = np.concatenate ((data_dict[location]['new_case'], [new_case]))
        data_dict[location]['new_death'] = np.concatenate ((data_dict[location]['new_death'], [new_death]))
        data_dict[location]['icu'] = np.concatenate ((data_dict[location]['icu'], [icu]))
        data_dict[location]['hosp_patients'] = np.concatenate ((data_dict[location]['hosp_patients'], [hosp_patients]))
        
    else:
        payload = {}
        payload['location'] = location
        payload['dates'] = np.array([date])
        payload['new_case'] = np.array([new_case])
        payload['new_death'] = np.array([new_death])
        payload['icu'] = np.array([icu])
        payload['hosp_patients'] = np.array([hosp_patients])
        
        payload['population'] = population
        payload['median_age'] = median_age
        payload['aged_65_older'] = aged_65_older
        payload['gdp_per_capita'] = gdp_per_capita
        payload['hospital_beds_per_thousand'] = hospital_beds_per_thousand
        
        data_dict[location] = payload
        


#print(data_dict)


# save processed data
loc_name = [k for k, _ in data_dict.items()]
data_field = [v for _, v in data_dict.items()]

# np.savez('data/covid_data.npz', loc_name=loc_name, data=data_field)
