def min_max_slice(tab):
    return min(tab), max(tab)

def mean_slice(tab):
    val_temp = 0
    for score in tab:
       val_temp += score
    return val_temp / len(tab)

def get_mean(tab, slices):
    tab_max = []
    tab_min = []
    mean_slices = []

    for i in range(0, (len(tab)/slices)):
        tranche = []
        for index in range(0,slices):
            tranche.append(tab[index + i * slices])
        temp_min, temp_max = min_max_slice(tranche)
        temp_mean = mean_slice(tranche, slices)
        tab_min.append(temp_min)
        tab_max.append(temp_max)
        mean_slices.append(temp_mean)

    overall_max = max(tab_max)
    overall_min = min(tab_min)
    overall_mean = mean_slice(mean_slices)
    
    return tab_max, tab_min, mean_slices, overall_max, overall_min, overall_mean
            
        