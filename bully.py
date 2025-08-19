def bully_algorithm(processes, initiator):
    n = len(processes)
    coordinator = max(processes)  # highest id process is coordinator
    print(f"Process {initiator} initiates election.")

    for p in sorted(processes, reverse=True):
        if p > initiator:
            print(f"Process {p} responds to election.")
            return bully_algorithm(processes, p)
    print(f"Process {initiator} becomes coordinator.")
    return initiator

# Example usage:
processes = [1, 2, 3, 4, 5]
coordinator = bully_algorithm(processes, initiator=2)
print(f"Coordinator is process {coordinator}")
