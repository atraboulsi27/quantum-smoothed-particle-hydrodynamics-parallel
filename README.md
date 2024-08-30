QuantumSPH Parallelized Version
Overview
This repository contains a performance modeling, OpenMP, and MPI parallelized version of the original QuantumSPH project by Philip Mocz (2017) from Harvard University. Our enhanced version leverages parallel computing to improve the efficiency and scalability of the Schrödinger Equation Smoothed Particle Hydrodynamics (SPH) method solver.

Original Project
The original QuantumSPH project is a minimal working example demonstrating the concept presented in Mocz & Succi (2015). It solves the Schrödinger equation for a particle in a Simple Harmonic Oscillator (SHO) potential.

DD2356 VT24 Methods in High Performance Computing
Quantum Smoothed-Particle
Hydrodynamics
Ahmad Traboulsi Sergiu Bogdan Popescu

Experimental Setup

For implementing and testing this project we used both our local machines and the Dardel
Supercomputer.
Since Dardel has been offline for more than a week, we could not run it to validate or
collect data. Thus, for the OpenMP implementation, the results presented in this report are acquired
on our local machines. By the time we started the MPI implementation, Dardel was back online,
therefore the results presented in the MPI section are gathered on the supercomputer.

Introduction

In the realm of computational quantum mechanics, efficient and precise simulation
methods are paramount for understanding and predicting quantum systems' behavior. One such
method is Quantum Smoothed-Particle Hydrodynamics (QSPH), a computational technique that
models quantum mechanical phenomena using a smoothed-particle approach.
We based our project work on a popular Python implementation for QSPH, developed by
Philip Mocz and available in the repository QuantumSPH, as it serves as a robust foundation for
these simulations. However, Python, while versatile and user-friendly, may not always provide the
computational efficiency required for large-scale or highly complex simulations. Therefore,
porting this code to a lower-level language like C, known for its performance efficiency, becomes
essential.
The primary aim of this report is to document the process of porting the QSPH Python code
to C, and further optimizing it through parallelization using OpenMP and MPI. This transformation
is motivated by the need to enhance computational performance, enabling the handling of more
extensive and intricate quantum systems.
Overview of Quantum Smoothed-Particle Hydrodynamics (QSPH)
QSPH is a numerical technique that blends principles of quantum mechanics with the
computational framework of smoothed-particle hydrodynamics (SPH). In essence, it represents
quantum particles as smoothed particles, facilitating the simulation of their collective behavior 
over time. This method is particularly useful in scenarios where quantum effects significantly
influence the system's dynamics, such as in condensed matter physics, astrophysics, and material
science.
The Python code from the QuantumSPH repository implements this method by solving the
Schrödinger equation for a system of particles. The script discretizes the wave function and evolves
it over time using the smoothed-particle approach, allowing for the modeling of complex quantum
phenomena with relative ease. Despite its effectiveness, the Python implementation has limitations
regarding computational speed and scalability, especially when dealing with large-scale
simulations.

Motivation for Porting to C

C is a widely-used programming language known for its execution speed and low-level
memory management capabilities. By porting the QSPH code from Python to C, we aim to achieve
several key improvements:
1. Performance Enhancement: C programs typically execute faster than Python scripts due
to the compiled nature of C and its efficient use of system resources.
2. Memory Management: C provides greater control over memory allocation and
management, which can be crucial for handling the large datasets often involved in
quantum simulations.
3. Scalability: The ability to manage and optimize memory usage in C facilitates the handling
of larger and more complex simulations, improving scalability.
Parallelization with OpenMP and MPI
To further enhance the computational efficiency, we parallelized the C implementation using two
widely-used parallel computing libraries: OpenMP and MPI.
1. OpenMP: OpenMP is an API that supports multi-platform shared memory
multiprocessing programming in C. It allows the easy creation of multi-threaded
applications, enabling concurrent execution of code segments. By parallelizing the QSPH
computations with OpenMP, we can significantly reduce the execution time on multi-core
processors.
2. MPI: The Message Passing Interface (MPI) is a standardized and portable messagepassing system designed to function on a wide variety of parallel computing architectures.
Unlike OpenMP, which is best suited for shared memory systems, MPI excels in distributed
memory environments. Implementing MPI in the QSPH code allows it to run on distributed
computing systems, such as clusters, thus facilitating the handling of even larger
simulations by distributing the workload across multiple nodes.

In summary, porting the QSPH code from Python to C and parallelizing it with OpenMP and
MPI not only enhances performance but also expands the capability to handle more complex and
larger-scale quantum simulations. This report will delve into the detailed steps of this porting and 
parallelization process, highlighting the challenges encountered and the solutions employed to
achieve the desired computational efficiency.

Output of Initial Script in Python

As presented in the introduction of this report, the initial script provides a versatile way of
simulating and also calculating the Quantum Smoothed-Particle Hydrodynamics. The output of
this script represents a series of data that can be plotted to show the probability distribution at
different timestamps of a particle to be at a certain position.

![image](https://github.com/user-attachments/assets/d9e6b0fb-4734-4800-ab3c-f0784a39acb9)

As it can be observed in the previous image the output of the script provides the results
both from the simulation (colored lines) and from the exact mathematical equation (grey lines)
used to simulate the quantum behavior.

Porting the Code from C to Python

Before starting the porting process, we need to understand the main and most difficult areas
of interest in the initial script. The algorithm is composed of 2 main parts:

 A main loop that based on different physical properties, computes the acceleration
of the particle.

 Multiple computations responsible for calculating the physical properties (ex:
acceleration, pressure).

After understanding the overall structure of the Python script, most of the porting process
is straight forward. We followed the same structure and as for the data structures we used
dynamically allocated arrays that we passed to different functions, thus making sure the changes
in data are preserved. The part where we difficulties was the one where we had to take into account
the implementation of the build in capabilities for matrix multiplication that are not present in C.
For simpler explanation we will show how we implemented the Density function in C, as it
provides a standard structure that is also used in the other functions. 

![image](https://github.com/user-attachments/assets/54c569cf-8971-4018-8f0f-913aa45568c2)

Validation of Results

For validating the results, we saved the results generated by both Python and C scripts in
different output files and using a different Python script we validated the data in 2 ways:
 Visual validation: we plotted both results and made sure the same distributions
across time and space are present.

 Arithmetic validation: we computed the MSE between the 2 datasets and verified
it to be zero.

![image](https://github.com/user-attachments/assets/8d89e29f-0b98-4895-8669-feb7f07d3832)

Serial code analysis and timing

Before testing the performance of the serial code and the parallel code, we increased the
simulation steps from the original 100 in the raw script to 10000 in our ported and parallel scripts.
This is in order to aid us in having more insights and more direct comparisons when we want to
test. We used the Gprof profiler and we only analyzed the hierarchical mode, as it provides a tree
like structure of the function calls and their time of execution.

From the output of the timing analysis, we can see that the top bottlenecks, in order, would
be the pressure, acceleration and density functions, apart form the bottleneck present in the main
function (we cannot improve this one, as each loop iteration is dependent on previous iterations)
These bottlenecks will be our focus when we optimize using parallelism in the later slides.

![image](https://github.com/user-attachments/assets/298e34bd-cbad-4e69-8915-23d34fd09509)

As for the execution time, the serial implementation in C took an average of 10.81 seconds,
in which the processor the was used for testing had 4.69 instruction per cycle, with a branch
miss percentage of 0.33%.
OpenMP parallel implementation
In the modified script which can be found on Github, OpenMP is utilized to parallelize
multiple computationally intensive sections, thereby enhancing performance by leveraging multicore processors. The #pragma omp parallel for directives are strategically placed before loops
in functions such as linspace, density, pressure, acceleration, probeDensity, and
probeDensityExact.
Additionally, omp_set_num_threads(16) sets the number of threads to 16, ensuring that
the workload is distributed across 16 threads for optimal performance. The use of private variables
within parallel regions prevents race conditions, ensuring accurate and independent computation
by each thread.
After multiple tests with different OpenMP approaches, we have decided to drop the
approach where we used to parallelize a nested for loop in favor of just parallelizing on the upper
level since after testing, we found a huge performance drop that affected the performance in a way
where the increase of number of threads lead to increase in time. We theorize that this increase
was due to the larger overhead of dealing with nested parallel loops and that delegating the second
loop to each thread had more optimal execution even if it is possible to make it more parallel. 

![image](https://github.com/user-attachments/assets/134d5750-ff2d-474f-84ec-51df2499fff7)

Regarding the timing analysis we got the following results using the GProf profiler:
 Total execution time: 2.279 seconds (best result is reached at 16 threads)
 Disappearance of bottlenecks in the auxiliary functions
 Instructions per Cycle: 1.54
 Cache misses: 0.24%
As it can be observed from the above graph, the OpenMP greatly reduces the overall
execution time.
MPI parallel implementation
In this implementation, the Message Passing Interface (MPI) is used to parallelize the
calculations for a QuantumSPH simulation. MPI enables the distribution of computational tasks
across multiple processors, enhancing the efficiency and scalability of the simulation. The code
uses MPI to distribute arrays and workload among the processes for key functions like computing
particle densities, pressures, and accelerations.
For instance, the linspace function divides the range into segments handled by different
processes, gathering results with MPI_Gatherv. Similarly, the density, pressure, acceleration,
probeDensity, and probeDensityExact functions perform local calculations on subsets of data
and aggregate results using MPI collective operations. This parallel approach allows the simulation
to handle larger datasets and more complex computations within a reasonable time frame,
leveraging the power of distributed memory systems. 

Regarding the timing analysis we got the following results using the GProf profiler:

 Total execution time: 4.43 seconds (best time is reached for 32 threads)

 Disappearance of bottlenecks in the auxiliary functions

 Instructions per Cycle: 2.36

 Cache misses: 13.6%
As it can be observed from the above graph, the MPI usage greatly reduces the overall
execution time. It is worth mentioning that the overall performance is better when using
OpenMP, and we theorize this fact is because of the higher overhead of the MPI library (this can
be avoided if we use bigger datasets). 
References
 https://github.com/pmocz/QuantumSPH
 Methods in High Performance Computing Lectures (Module 1, 2 & 3) 
