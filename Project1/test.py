import random


def tournament_selection(a, k):
    count = 0
    bestParent = -9999999999999.0
    selected = []
    best_index = 0
    

    
    t = a[:]
    population_size = len(t)
    for i in range(k):
        index = random.randint(0,population_size -1)
        thisParent = t[index]
        selected.append(thisParent)
        if thisParent > bestParent:
            bestParent = thisParent
        t.remove(t[index])
        population_size = len(t)


    print("K selected  = ", selected)

    best_index = a.index(bestParent)
    print("Best Index =  ", best_index)
    print("Best Parent = ", bestParent)
    return best_index

a = [23, 24, 5, 4, 32, 999, 45, 43, 7]
print("The population = ", a)
print("number of individuals = ", len(a))

print(tournament_selection(a, 4))