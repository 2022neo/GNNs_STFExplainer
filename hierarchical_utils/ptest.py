import scipy.stats as stats

baseline_data = [
]  
stf_data = [
]  

#  Mann-Whitney U
statistic, p_value = stats.mannwhitneyu(stf_data, baseline_data, alternative='two-sided')
print(statistic)
print(p_value)
