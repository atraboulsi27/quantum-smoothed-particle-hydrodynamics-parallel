import matplotlib.pyplot as plt

# Define the number of threads and corresponding execution times
threads = [2, 4, 8, 16, 32, 64, 128]
execution_times = [20.9, 11.04, 6.39, 4.46, 4.45, 4.41, 5.28]

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
