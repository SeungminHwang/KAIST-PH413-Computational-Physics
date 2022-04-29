import datetime as dt
import numpy as np
import matplotlib.pyplot as plt


covid_data = np.load('data/covid_data.npz', allow_pickle=True)

country_names = covid_data['loc_name']
data = covid_data['data']



# construct map for country names
country_idx_map = {}
for i, name in enumerate(country_names):
    country_idx_map[name] = i
#print(country_idx_map)




target_countries = ['South Korea', 'United States', 'Germany']

ax = plt.subplot()

for name in target_countries:
    idx = country_idx_map[name]
    
    country_data = data[idx]
    
    country_date = country_data['dates']
    country_positive = country_data['new_case']
    country_death = country_data['new_death']
    
    
    x_values = [dt.datetime.strptime(d,"%Y-%m-%d").date() for d in country_date]
    ax.plot(x_values, country_positive)

    
ax.legend(target_countries)
plt.show()



