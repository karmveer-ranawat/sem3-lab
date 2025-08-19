def bankers_algorithm(available, max_demand, allocation):
    need = [[max_demand[i][j] - allocation[i][j] for j in range(len(available))] for i in range(len(allocation))]
    finish = [False] * len(allocation)
    safe_seq = []
    while len(safe_seq) < len(allocation):
        allocated = False
        for i in range(len(allocation)):
            if not finish[i] and all(need[i][j] <= available[j] for j in range(len(available))):
                for j in range(len(available)):
                    available[j] += allocation[i][j]
                finish[i] = True
                safe_seq.append(i)
                allocated = True
        if not allocated:
            return False, []  # No safe sequence found
    return True, safe_seq

# Example usage:
available = [3, 3, 2]
max_demand = [[7, 5, 3], [3, 2, 2], [9, 0, 2], [2, 2, 2], [4, 3, 3]]
allocation = [[0, 1, 0], [2, 0, 0], [3, 0, 2], [2, 1, 1], [0, 0, 2]]

safe, seq = bankers_algorithm(available, max_demand, allocation)
print("Safe Sequence:" if safe else "No safe sequence", seq)
