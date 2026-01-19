def min_max_slice(tab):
    return min(tab), max(tab)

def mean_slice(tab):
    return sum(tab) / len(tab)

def rounding(tab_nbr):
    return [round(nbr, 3) for nbr in tab_nbr]

def get_mean(tab, slices):
    tab_max = []
    tab_min = []
    mean_slices = []

    # Tranches complètes
    for i in range(0, len(tab) // slices):
        tranche = []
        for index in range(slices):
            tranche.append(tab[index + i * slices])

        temp_min, temp_max = min_max_slice(tranche)
        temp_mean = mean_slice(tranche)

        tab_min.append(temp_min)
        tab_max.append(temp_max)
        mean_slices.append(temp_mean)

    # Dernière tranche (reste)
    start = (len(tab) // slices) * slices
    if start < len(tab):
        tranche = tab[start:len(tab)]

        temp_min, temp_max = min_max_slice(tranche)
        temp_mean = mean_slice(tranche)

        tab_min.append(temp_min)
        tab_max.append(temp_max)
        mean_slices.append(temp_mean)

    # Stats globales (sur les vraies données)
    overall_max = max(tab)
    overall_min = min(tab)
    overall_mean = mean_slice(tab)

    tab_max_round = rounding(tab_max)
    tab_min_round = rounding(tab_min)
    mean_slices_round = rounding(mean_slices)

    return tab_max_round, tab_min_round, mean_slices_round, round(overall_max), round(overall_min), round(overall_mean)
