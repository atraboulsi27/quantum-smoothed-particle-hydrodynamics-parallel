QuantumSPH Parallelized Version
Overview
This repository contains a performance modeling, OpenMP, and MPI parallelized version of the original QuantumSPH project by Philip Mocz (2017) from Harvard University. Our enhanced version leverages parallel computing to improve the efficiency and scalability of the Schrödinger Equation Smoothed Particle Hydrodynamics (SPH) method solver.

Original Project
The original QuantumSPH project is a minimal working example demonstrating the concept presented in Mocz & Succi (2015). It solves the Schrödinger equation for a particle in a Simple Harmonic Oscillator (SHO) potential.

Enhancements
Our contributions to the original project include:

Performance Modeling: Detailed analysis and optimization of the solver's performance.
OpenMP Parallelization: Parallelized computation within a single machine using OpenMP.
MPI Parallelization: Distributed computation across multiple machines using MPI.
