from collections import defaultdict

# Example list of tuples
data = [('A', 10.5), ('B', 20), ('A', 30), ('B', 40), ('A', 50)]

# Dictionary to store sum and count per class
class_sums = defaultdict(float)
class_counts = defaultdict(float)

# Aggregate sums and counts
for cls, value in data:
    class_sums[cls] += value
    class_counts[cls] += 1

# Calculate averages
class_averages = {cls: class_sums[cls] / class_counts[cls] for cls in class_sums}

print(class_averages)


A = "300_1"

print(A.isnumeric())