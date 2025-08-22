import threading

data = ["hello world", "hello again", "hello world again"]

mapped = []
lock = threading.Lock()

def map_function(text):
    local = []
    words = text.split()
    for word in words:
        local.append((word, 1))
    with lock:
        mapped.extend(local)

threads = []
for line in data:
    t = threading.Thread(target=map_function, args=(line,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

print("Mapped Output:", mapped)

shuffle = {}
for word, count in mapped:
    if word in shuffle:
        shuffle[word].append(count)
    else:
        shuffle[word] = [count]

reduced = {}
for word, counts in shuffle.items():
    reduced[word] = sum(counts)

print("\nFinal Word Count:", reduced)
