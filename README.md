# Quick Description

**Programmers :shipit:**: Zakaria Abdelmoiz DAHI from the University of Malaga, Spain. 

**About:** This repositiory implements an evolutonary algorithm devised in [1] for optimising qubits' initialisation in IBM machines.

- [1] Z.A DAHI, F. Chicano, G. Luque, and E. Alba. 2022. Genetic algorithm for qubits initialisation in noisy intermediate-scale quantum machines: the IBM case study. In Proceedings of the Genetic and Evolutionary Computation Conference (GECCO '22). Association for Computing Machinery, New York, NY, USA, 1164–1172. https://doi.org/10.1145/3512290.3528830

## **How :green_book:** 

- In the folder `code`, you can find folders having as name the percentage of entanglement `0.25`, `0.50`, ...
- Choose one of them. In each fo them, you will find a folder having as name the number of qubits in the used machine, `7`, `16`, `20`, ...
- Each subfolder contains two files and one folder:
    - Results: contains the results of the experiments.
    - `main.py`: is the code of the devised approach
    - `launch.sh`: is dedicated to slurm-based cluster execution.
    - If you want to run some code, just run its corresponding `main.py`.


## **Folders Hiearchy :open_file_folder:**
    
- Code:
    - `0.25, 0.50, 0.75, 1.00`: contains the code when using the corresponding entanglement percentage:
        - `7, 16, 20, 20, 27, 53, 65`: contains the code according to the number of the qubits in the machine.
        
## **Demo :movie_camera:**
    
- Please refer to the original paper [HERE](https://dl.acm.org/doi/abs/10.1145/3512290.3528830) for more detailed results and discussions.
