import matplotlib.pyplot as plt

rr_list = []
rr_exact_list = []

def read_C_data_into_list():
    filename = r'/home/atrab/Desktop/repos/quantum-sph-parallel/HPC_Project-master/results.txt'
    global rr_list
    try:
        # Open the file in read mode
        with open(filename, 'r') as file:
            lines = file.readlines()
            # Read all lines from the file
            counter = 0;
            for line in lines:
                if counter % 2 == 0:
                    array = [float(value) for value in line.split()]
                    rr_list.append(array)
                else:
                    array = [float(value) for value in line.split()]
                    rr_exact_list.append(array)
                counter = counter + 1

    except Exception as e:
        print(f"An error occurred: {e}")


def read_python_data_into_list():
    filename = r'C:\Users\sergi\source\repos\High_Performance_Computing\High_Performance_Computing\results_python.txt'
    global results_python
    try:
        # Open the file in read mode
        with open(filename, 'r') as file:
            # Read all lines from the file
            lines = file.readlines()
            # Convert each line to a float and add to the list
            results_python = [float(line.strip()) for line in lines]
    except Exception as e:
        print(f"An error occurred: {e}")


read_C_data_into_list()
# read_python_data_into_list()

print(len(rr_list))

# plt.plot(results_python, marker='o', linestyle='-', color='b', label='python')
for i in range(4):
    plt.plot(rr_list[i], linewidth=2, color=[1. * (i+1) * 25 / 100, 0, 1. - 1. * (i+1)*25 / 100], label=i)

for i in range(4):
    plt.plot(rr_exact_list[i], linewidth=2, color=[.6, .6, .6])

# Adding title and labels
plt.title('Results based on C code')

# Adding a legend
plt.legend()

# Show the plot
plt.show()

