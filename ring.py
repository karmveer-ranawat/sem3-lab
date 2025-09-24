import random
import time
from threading import Thread, Lock

# Number of processes in the ring
num_processes = 5

class Process:
    def __init__(self, pid):
        self.pid = pid
        self.is_leader = False
        self.lock = Lock()

    def run(self):
        # Simulate some work
        time.sleep(random.uniform(0.1, 0.5))

        # Pass a message to the next process in the ring
        next_process = (self.pid + 1) % num_processes
        with processes[next_process].lock:
            if not processes[next_process].is_leader:
                print(f"Process {self.pid} sends a message to Process {next_process}")

            # Check if we should become the leader
            if self.pid == max([p.pid for p in processes]):
                self.is_leader = True
                print(f"Process {self.pid} becomes the leader")

# Create a list of processes
processes = [Process(pid) for pid in range(num_processes)]

# Start threads for each process
threads = [Thread(target=process.run) for process in processes]
for thread in threads:
    thread.start()

# Wait for all threads to finish
for thread in threads:
    thread.join()

# Find the leader
leader = [p for p in processes if p.is_leader][0]
print(f"Leader elected: Process {leader.pid}")