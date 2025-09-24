import random

class Process:
    def __init__(self, pid):
        self.pid = pid
        self.active = True

    def __repr__(self):
        return f"P{self.pid}{' (inactive)' if not self.active else ''}"


def ring_election(processes, initiator_id):
    n = len(processes)
    initiator = processes[initiator_id]

    if not initiator.active:
        print(f"Process {initiator.pid} is inactive and cannot initiate election.")
        return None

    print(f"\nElection initiated by Process {initiator.pid}")

    # Election message will circulate the ring
    message = [initiator.pid]
    current = (initiator_id + 1) % n

    while current != initiator_id:
        if processes[current].active:
            message.append(processes[current].pid)
            print(f"Process {processes[current].pid} passes election message.")
        current = (current + 1) % n

    # The process with the highest ID wins
    leader = max(message)
    print(f"Processes in election: {message}")
    print(f"Leader elected: Process {leader}\n")

    return leader


# Example usage
if __name__ == "__main__":
    # Create 5 processes
    processes = [Process(i) for i in range(5)]

    # Randomly deactivate one process
    processes[random.randint(0, 4)].active = False

    print("Initial Processes:", processes)

    # Start an election from process 2
    leader = ring_election(processes, initiator_id=2)
