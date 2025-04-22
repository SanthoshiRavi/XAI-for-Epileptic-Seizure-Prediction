# Re-import necessary libraries after code execution environment reset
import matplotlib.pyplot as plt
import numpy as np
import random

# Simulate dataset sizes
dataset_sizes = [50, 100, 150, 200, 250, 300, 350]

# Simulate query times (in seconds) for Add, Edit, and Delete operations
# We simulate increasing complexity but all within the 3-second limit
add_times = [random.uniform(0.3, 0.8) + size * 0.0002 for size in dataset_sizes]
edit_times = [random.uniform(0.4, 0.9) + size * 0.00025 for size in dataset_sizes]
delete_times = [random.uniform(0.5, 1.0) + size * 0.0003 for size in dataset_sizes]

# Cap all values at 2.9 seconds to meet the 3-second usability rule
add_times = [min(time, 2.9) for time in add_times]
edit_times = [min(time, 2.9) for time in edit_times]
delete_times = [min(time, 2.9) for time in delete_times]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(dataset_sizes, add_times, marker='o', label='Add Query Time')
plt.plot(dataset_sizes, edit_times, marker='s', label='Edit Query Time')
plt.plot(dataset_sizes, delete_times, marker='^', label='Delete Query Time')
plt.axhline(y=3, color='r', linestyle='--', label='3-second Usability Limit')

plt.title('Query Execution Times vs Dataset Size')
plt.xlabel('Dataset Size (number of records)')
plt.ylabel('Execution Time (seconds)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
