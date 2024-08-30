import matplotlib.pyplot as plt

# Define the number of threads and corresponding execution times
threads = [1, 2, 4, 8, 16]
execution_times = [10.694, 5.531, 3.199, 3.229, 2.279]

# Plot the data
plt.figure(figsize=(8, 6))
plt.plot(threads, execution_times, marker='o', linestyle='-')
plt.title('Execution Time vs. Number of Threads')
plt.xlabel('Number of Threads')
plt.ylabel('Execution Time (seconds)')
plt.grid(True)
plt.xticks(threads)  # Ensure all threads are shown on the x-axis
plt.tight_layout()

# Show the plot
plt.show()
