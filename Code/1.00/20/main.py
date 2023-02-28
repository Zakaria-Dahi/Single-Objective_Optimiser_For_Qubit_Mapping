#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 15:49:17 2021

@author: user
FakeOpenPulse2Q => 2 qubits
FakeTenerife => 5 qubits
FakeMelbourne => 14 qubits
FakeRueschlikon => 16 qubits
FakePoughkeepsie => 20 qubits
FakeTokyo => 20 qubits
etc.
The above backends are the only one given in qiskit help
I have discovered the remaining one by writing Fake.. and let the spyder help suggest
There are 1 qubit backends but I neglected them
We use the mock because it is dedicated to testing the compiler. (e.g. Armonk)
nOTE: 'FakeMumbaiV2' HAS BEEN NEGLECTED FOR THE MOMENT.
It simulates a GhZ state
"""
import sys
import math
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_circuit_layout
from qiskit.test.mock import FakeYorktown,FakeVigo,FakeValencia,FakeToronto,FakeSydney,FakeSingapore,FakeSantiago,FakeRome,FakeRochester,FakeQuito,FakePoughkeepsie,FakeParis,FakeOurense,FakeOpenPulse3Q,FakeMumbaiV2,FakeMumbai,FakeMontreal,FakeManila,FakeManhattan,FakeLondon,FakeLima,FakeLegacyYorktown,FakeLegacyVigo,FakeLegacyValencia,FakeLegacyToronto,FakeLegacyTokyo,FakeLegacyTenerife,FakeLegacySydney,FakeLegacySingapore,FakeLegacySantiago,FakeLegacyRueschlikon,FakeLegacyRome,FakeLegacyRochester,FakeLegacyQuito,FakePoughkeepsie,FakeLegacyParis,FakeLegacyOurense,FakeLegacyMumbai,FakeLegacyMontreal,FakeLegacyMelbourne,FakeLegacyManhattan,FakeLegacyLondon,FakeLegacyLima,FakeLegacyJohannesburg,FakeLegacyEssex,FakeLegacyCasablanca,FakeLegacyCambridgeAlternativeBasis,FakeLegacyCambridge,FakeLegacyBurlington,FakeLegacyBogota,FakeLegacyBoeblingen,FakeLegacyBelem,FakeLegacyAthens,FakeLegacyAlmaden,FakeLagos,FakeJohannesburg,FakeJakarta,FakeGuadalupe,FakeEssex,FakeCasablanca,FakeCambridgeAlternativeBasis,FakeCambridge,FakeBurlington,FakeBrooklyn,FakeBogota,FakeBoeblingen,FakeBelem,FakeAthens,FakeAlmaden,FakeOpenPulse2Q, FakeTenerife, FakeMelbourne, FakeRueschlikon, FakePoughkeepsie, FakeTokyo
import numpy as np
import statistics
import random 
import matplotlib.pyplot as plt
import xlwt
import time


def test_backend(machine2):
    if machine2 == "FakeOpenPulse2Q":
        backend2 = FakeOpenPulse2Q()  
    if machine2 == "FakeTenerife":
        backend2 = FakeTenerife()
    if machine2 == "FakeMelbourne":
        backend2 = FakeMelbourne()
    if machine2 == "FakeRueschlikon":
        backend2 = FakeRueschlikon()
    if machine2 == "FakePoughkeepsie":
        backend2 = FakePoughkeepsie()
    if machine2 == "FakeTokyo":
        backend2 = FakeTokyo()
    if machine2 == "FakeAlmaden":
        backend2 = FakeAlmaden()          
    if machine2 == "FakeAthens":
        backend2 = FakeAthens()  
    if machine2 == "FakeBelem":
        backend2 = FakeBelem()  
    if machine2 == "FakeBoeblingen":
        backend2 = FakeBoeblingen()
    if machine2 == "FakeBogota":
        backend2 = FakeBogota()
    if machine2 == "FakeBrooklyn":
        backend2 = FakeBrooklyn()  
    if machine2 == "FakeBurlington":
        backend2 = FakeBurlington()
    if machine2 == "FakeCambridge":
        backend2 = FakeCambridge()
    if machine2 == "FakeCambridgeAlternativeBasis":
        backend2 = FakeCambridgeAlternativeBasis()  
    if machine2 == "FakeCasablanca":
        backend2 = FakeCasablanca()
    if machine2 == "FakeEssex":
        backend2 = FakeEssex()
    if machine2 == "FakeGuadalupe":
        backend2 = FakeGuadalupe()  
    if machine2 == "FakeJakarta":
        backend2 = FakeJakarta()
    if machine2 == "FakeJohannesburg":
        backend2 = FakeJohannesburg()
    if machine2 == "FakeLagos":
        backend2 = FakeLagos()  
    if machine2 == "FakeLegacyAlmaden":
        backend2 = FakeLegacyAlmaden()
    if machine2 == "FakeLegacyAthens":
        backend2 = FakeLegacyAthens()
    if machine2 == "FakeLegacyBelem":
        backend2 = FakeLegacyBelem()  
    if machine2 == "FakeLegacyBoeblingen":
        backend2 = FakeLegacyBoeblingen()
    if machine2 == "FakeLegacyBogota":
        backend2 = FakeLegacyBogota()            
    if machine2 == "FakeLegacyBurlington":
        backend2 = FakeLegacyBurlington()
    if machine2 == "FakeLegacyCambridge":
        backend2 = FakeLegacyCambridge()  
    if machine2 == "FakeLegacyCambridgeAlternativeBasis":
        backend2 = FakeLegacyCambridgeAlternativeBasis()
    if machine2 == "FakeLegacyCasablanca":
        backend2 = FakeLegacyCasablanca()
    if machine2 == "FakeLegacyEssex":
        backend2 = FakeLegacyEssex()  
    if machine2 == "FakeLegacyJohannesburg":
        backend2 = FakeLegacyJohannesburg()
    if machine2 == "FakeLegacyLima":
        backend2 = FakeLegacyLima()
    if machine2 == "FakeLegacyLondon":
        backend2 = FakeLegacyLondon()  
    if machine2 == "FakeLegacyManhattan":
        backend2 = FakeLegacyManhattan()
    if machine2 == "FakeLegacyMelbourne":
        backend2 = FakeLegacyMelbourne() 
    if machine2 == "FakeLegacyMontreal":
        backend2 = FakeLegacyMontreal()
    if machine2 == "FakeLegacyMumbai":
        backend2 = FakeLegacyMumbai()
    if machine2 == "FakeLegacyOurense":
        backend2 = FakeLegacyOurense()  
    if machine2 == "FakeLegacyParis":
        backend2 = FakeLegacyParis()
    if machine2 == "FakePoughkeepsie":
        backend2 = FakePoughkeepsie() 
    if machine2 == "FakeLegacyQuito":
        backend2 = FakeLegacyQuito()
    if machine2 == "FakeLegacyRochester":
        backend2 = FakeLegacyRochester()
    if machine2 == "FakeLegacyRome":
        backend2 = FakeLegacyRome()  
    if machine2 == "FakeLegacyRueschlikon":
        backend2 = FakeLegacyRueschlikon()
    if machine2 == "FakeLegacySantiago":
        backend2 = FakeLegacySantiago() 
    if machine2 == "FakeLegacySingapore":
        backend2 = FakeLegacySingapore()  
    if machine2 == "FakeLegacySydney":
        backend2 = FakeLegacySydney()
    if machine2 == "FakeLegacyTenerife":
        backend2 = FakeLegacyTenerife() 
    if machine2 == "FakeLegacyTokyo":
        backend2 = FakeLegacyTokyo()
    if machine2 == "FakeLegacyToronto":
        backend2 = FakeLegacyToronto()
    if machine2 == "FakeLegacyValencia":
        backend2 = FakeLegacyValencia()  
    if machine2 == "FakeLegacyVigo":
        backend2 = FakeLegacyVigo()
    if machine2 == "FakeLegacyYorktown":
        backend2 = FakeLegacyYorktown() 
    if machine2 == "FakeLima":
        backend2 = FakeLima()
    if machine2 == "FakeLondon":
        backend2 = FakeLondon()
    if machine2 == "FakeManhattan":
        backend2 = FakeManhattan()  
    if machine2 == "FakeManila":
        backend2 = FakeManila()
    if machine2 == "FakeMontreal":
        backend2 = FakeMontreal() 
    if machine2 == "FakeMumbai":
        backend2 = FakeMumbai()
    if machine2 == "FakeMumbaiV2":
        backend2 = FakeMumbaiV2()
    if machine2 == "FakeOpenPulse3Q":
        backend2 = FakeOpenPulse3Q()  
    if machine2 == "FakeOurense":
        backend2 = FakeOurense()
    if machine2 == "FakeParis":
        backend2 = FakeParis()             
    if machine2 == "FakePoughkeepsie":
        backend2 = FakePoughkeepsie() 
    if machine2 == "FakeQuito":
        backend2 = FakeQuito()
    if machine2 == "FakeRochester":
        backend2 = FakeRochester()
    if machine2 == "FakeRome":
        backend2 = FakeRome()  
    if machine2 == "FakeSantiago":
        backend2 = FakeSantiago()
    if machine2 == "FakeSingapore":
        backend2 = FakeSingapore()              
    if machine2 == "FakeSydney":
        backend2 = FakeSydney() 
    if machine2 == "FakeToronto":
        backend2 = FakeToronto()
    if machine2 == "FakeValencia":
        backend2 = FakeValencia()
    if machine2 == "FakeVigo":
        backend2 = FakeVigo()  
    if machine2 == "FakeYorktown":
        backend2 = FakeYorktown()           
    return backend2;


def machine_selection(lb):
    backend_lists = ['FakeYorktown','FakeVigo','FakeValencia','FakeToronto','FakeSydney','FakeSingapore','FakeSantiago','FakeRome','FakeRochester','FakeQuito','FakePoughkeepsie','FakeParis','FakeOurense','FakeOpenPulse3Q','FakeMumbai','FakeMontreal','FakeManila','FakeManhattan','FakeLondon','FakeLima','FakeLegacyYorktown','FakeLegacyVigo','FakeLegacyValencia','FakeLegacyToronto','FakeLegacyTokyo','FakeLegacyTenerife','FakeLegacySydney','FakeLegacySingapore','FakeLegacySantiago','FakeLegacyRueschlikon','FakeLegacyRome','FakeLegacyRochester','FakeLegacyQuito','FakePoughkeepsie','FakeLegacyParis','FakeLegacyOurense','FakeLegacyMumbai','FakeLegacyMontreal','FakeLegacyMelbourne','FakeLegacyManhattan','FakeLegacyLondon','FakeLegacyLima','FakeLegacyJohannesburg','FakeLegacyEssex','FakeLegacyCasablanca','FakeLegacyCambridgeAlternativeBasis','FakeLegacyCambridge','FakeLegacyBurlington','FakeLegacyBogota','FakeLegacyBoeblingen','FakeLegacyBelem','FakeLegacyAthens','FakeLegacyAlmaden','FakeLagos','FakeJohannesburg','FakeJakarta','FakeGuadalupe','FakeEssex','FakeOpenPulse2Q','FakeTenerife','FakeMelbourne','FakeRueschlikon','FakePoughkeepsie','FakeTokyo','FakeAlmaden','FakeAthens','FakeBelem','FakeBoeblingen','FakeBogota','FakeBrooklyn','FakeBurlington','FakeCambridge','FakeCambridgeAlternativeBasis','FakeCasablanca']    
    configs = []
    for machine in backend_lists:
        backend = test_backend(machine)
        # recover the number of qubits in the machine
        qubits_num = backend.configuration().n_qubits;    
        configs.append(qubits_num);
    # extract the classes of machines and the number of qubits they have
    configs = np.copy(configs)
    configs_qubits = np.unique(configs)
    configs_qubits_sorted = np.sort(configs_qubits)
    backend_config = []
    # browse results from smallest number of qubits to the largest
    for i in range(len(configs_qubits_sorted)):
        if (configs_qubits_sorted[i]>=lb):
            for machine1 in backend_lists:
                backend1 = test_backend(machine1)
                num_qubits_processed = backend1.configuration().n_qubits
                if (configs_qubits_sorted[i] == num_qubits_processed):
                    backend_config.append(machine1)
    return backend_config;

def plot_results(iter_sample,e):
    plt.plot(iter_sample)
    plt.ylabel('Circuit depth')
    name = "GA_optimiser_exe_" + str(e+1) + ".png" 
    plt.savefig(name) 
    return;

def repair(individual,val_check,test2,dim):
    # start the reparation process
    qubits, counts = np.unique(individual, return_counts=True) 
    indices = np.where(counts>1)
    indices = indices[0]
    index_insert = 0
    for i in range(indices.size):
        double = qubits[indices[i]]
        pos = np.where(individual == double)
        pos = pos[0]
        if index_insert <= test2.size:
            for j in range(pos.size-1):
                individual[pos[j]] = test2[index_insert]
                index_insert= index_insert+1
    # test if the reparation has been done correctly                
    test2 = np.isin(val_check, individual) # check if the processed chromosome is representing all the clusters
    test2 =  val_check[test2 ==  False] # retrive the clusters that are not represented by the processed chromosome
    if np.size(test2) != 0: # ensures that there is a real lack of clusters before reparing the chromosome
        individual = repair(individual,val_check,test2,dim) # call the repair function
    return individual;

def layout_optimiser(iter,indiv,dim,pc,pm,shots,perc,e,backend,level):
    avg_fit_all = np.array([None]*iter)    
    worst_fit_all = np.array([None]*iter)    
    best_fit_all = np.array([None]*iter)
    best_fit  = math.inf;
    best_indiv = np.array([None]*dim)
    fitness_all = [None]*indiv
    fitness_all = np.array(fitness_all)
    # population initialisation
    pop = np.random.randint(0,dim,size=(indiv,dim))
    # ensure that all the qubits are encoded 
    val_check = np.arange(0,dim) # create a list of K evenly-spaced integers:s from 1 to dim qubits
    for i in range(indiv): # browse all the chromosomes in the population 
        test = np.isin(val_check, pop[i]) # check if the processed chromosome is representing all the qubits
        test2 =  val_check[test ==  False] # retrive the qubits that are not represented by the processed chromosome
        if np.size(test2) != 0: # ensures that there is a real lack of qubits before reparing the chromosome
           pop[i][0:dim] = repair(pop[i],val_check,test2,dim) # call the repair function
    # evaluate the initial population
    for i in range(indiv):
        fitness_all[i] =  fitness(pop[i],shots,backend,dim,level)

    # extract the best individual
    best_fit_aux  = np.amin(fitness_all) 
    if best_fit_aux <= best_fit:
        best_fit = best_fit_aux;
        index_best = np.where(fitness_all == best_fit)
        best_indiv = np.array(pop[index_best[0][0]],copy=True) # selected the first in case several chromosomes have the same ftness

    
    # enter in the main loop
    for i in range(iter):
        # selection
        mat_pool_size = int(math.ceil(perc * indiv)/100)
        parents = np.random.randint(0,indiv,size=(mat_pool_size,2))
        for off in range(mat_pool_size):
            parent1 = np.array(pop[parents[off,0]])
            parent2 = np.array(pop[parents[off,1]])
            offspring1 = np.array(parent1,copy=True) # just copy the first parent in the first offspring
            offspring2 = np.array(parent2,copy=True)  # just copy the second parent in the second offspring
            # crossover: two point from switchpoint[0] to (switch_point[1]-1)
            if random.random() < pc:
               switch_points = np.random.choice(np.arange(0,dim), replace=False, size=(2)) # generate the switch point where to apply the two-point crossover
               while switch_points[0] > switch_points[1]:
                    switch_points = np.random.choice(np.arange(0,dim), replace=False, size=(2)) 
               offspring1[switch_points[0]:switch_points[1]] = parent2[switch_points[0]:switch_points[1]] 
               offspring2[switch_points[0]:switch_points[1]] = parent1[switch_points[0]:switch_points[1]]  
            # mutation
            for index in range(dim):
                   if random.random() <= pm: # if condition of mutation satisfied perform the mutation
                       offspring1[index] = np.random.randint(0,dim) #apply the mutation           
            for index in range(dim):
                   if random.random() <= pm: # if condition of mutation satisfied perform the mutation
                       offspring2[index] = np.random.randint(0,dim) #apply the mutation           
            # repair offspring 1
            test = np.isin(val_check, offspring1) # check if the processed chromosome is representing all the qubits
            test2 =  val_check[test ==  False] # retrive the qubits that are not represented by the processed chromosome
            if np.size(test2) != 0: # ensures that there is a real lack of qubits before reparing the chromosome
                offspring1 = repair(offspring1,val_check,test2,dim) # call the repair function           
            # repair offspring 2
            test = np.isin(val_check, offspring2) # check if the processed chromosome is representing all the qubits
            test2 =  val_check[test ==  False] # retrive the qubits that are not represented by the processed chromosome
            if np.size(test2) != 0: # ensures that there is a real lack of qubits before reparing the chromosome
               offspring2 = repair(offspring2,val_check,test2,dim) # call the repair function                  
            # evaluation
            fitness1 =  fitness(offspring1,shots,backend,dim,level) 
            fitness2 =  fitness(offspring2,shots,backend,dim,level) 
            # replacement: apply (lambda, mu) replacement.
            worst_fit  = np.amax(fitness_all) 
            index_worst = np.where(fitness_all == worst_fit)
            if fitness1 <= worst_fit:
                fitness_all[index_worst] = fitness1
                pop[index_worst] = np.array(offspring1, copy = True)
                worst_fit  = np.amax(fitness_all) 
                index_worst = np.where(fitness_all == worst_fit)
                if fitness2 <= worst_fit:
                    fitness_all[index_worst] = fitness2
                    pop[index_worst] = np.array(offspring2, copy = True)
            else:     
                if fitness2 <= worst_fit:
                    fitness_all[index_worst] = fitness2
                    pop[index_worst] = np.array(offspring2, copy = True)

        # extract the best individual
        best_fit_aux  = np.amin(fitness_all) 
        if best_fit_aux <= best_fit:
            best_fit = best_fit_aux;
            index_best = np.where(fitness_all == best_fit)
            best_indiv = np.array(pop[index_best[0][0]],copy=True) # selected the first in case several chromosomes have the same ftness
        # print(best_fit) # display the best fitness in each iteration
        best_fit_all[i] = best_fit
        worst_fit_all[i] = np.amax(fitness_all) 
        avg_fit_all[i] = statistics.median(fitness_all)
    return best_fit,best_fit_all,best_indiv,worst_fit_all,avg_fit_all;
        

def fitness(individual,shots,backend,qubits_num,level):
    circuit_test = QuantumCircuit(qubits_num, qubits_num)
    circuit_test.h(0)
    circuit_test.cx(0,range(1,qubits_num))
    circuit_test.barrier()
    circuit_test.measure(range(qubits_num), range(qubits_num))
    sample_exec = [None]*shots; # store the circuit depth after each compilation 
    for sh in range(shots):
        # circuit_test.draw(output='mpl') # to plot the circuit
        compiled_circuit = transpile(circuit_test, backend, initial_layout=individual, optimization_level=level)
        # plot_circuit_layout(compiled_circuit, backend) # to plot the topology
        sample_exec[sh] = compiled_circuit.depth()
    return(statistics.median(sample_exec))


def fitness2(level,shots,backend,qubits_num):
    circuit_test = QuantumCircuit(qubits_num, qubits_num)
    circuit_test.h(0)
    circuit_test.cx(0,range(1,qubits_num))
    circuit_test.barrier()
    circuit_test.measure(range(qubits_num), range(qubits_num))
    sample_exec = [None]*shots; # store the circuit depth after each compilation 
    maxdepth = float('inf') # define and exagerated infinity
    for sh in range(shots):
        # circuit_test.draw(output='mpl') # to plot the circuit
        compiled_circuit = transpile(circuit_test, backend, optimization_level=level)
        # plot_circuit_layout(compiled_circuit, backend) # to plot the topology
        sample_exec[sh] = compiled_circuit.depth()
        if sample_exec[sh] < maxdepth:
            maxdepth = sample_exec[sh]
            layout_best = compiled_circuit._layout.get_physical_bits().items()
            layout_best_config = []
            for key, val in layout_best:
                layout_best_config.append(key)
    layout_best_final = layout_best_config[:qubits_num] # save only the qubits_num first, in case the circuit contains ancilla
    return(statistics.median(sample_exec)),layout_best_final

def main():
    """
    Circuit parameters:
    """
    lb = 7 # set the number of qubits that the machine should have
    shots = 30; # number of times the quantum circuit is executed.
    level = 3 # optimisation level
    # backend_config = machine_selection(lb)
    backend_config = ['FakeSingapore','FakePoughkeepsie','FakeLegacyTokyo','FakeLegacySingapore']
    for machine in backend_config:
        backend = test_backend(machine)
        # recover the number of qubits in the machine
        qubits_num = backend.configuration().n_qubits;    
        print(" ------------------------------ ")
        print("Experiments using the machine: "+ str(backend))
        print(" ------------------------------ ")          
        """
        Execute the GA Optimiser
        """
        print("  Step 1 using our Optimisation Routine ") 
        exe = 1; # number of executions
        iter = 30; # number of iterations the algorithm will perform
        indiv = 40; # number of chromosomes in the population
        dim = qubits_num; # size of the harware topology  = size of the chromosome
        pc = 0.5; # crossover probability
        pm = 0.1; # mutation probability
        perc = 50; # percentage of parents selected to create mating pool
        execution_1 = []; # stock the depth of the circuits obtained our routine
        iter_exe = np.array([0]*iter); # stock the best fitness in each iteration of all the executions
        iter_worst = np.array([0]*iter); # stock the worst fitness in each iterations of all the executions        
        iter_median = np.array([0]*iter); # stock the meadian population fitness in each iteration of all the executions                
        time1 = [] # stores the execution time of all the executions
        best_indiv_mat1 = np.array([0]*dim); # store the best individuals in each execution
        for e in range(exe):
             start =  time.time()
             exe_result_1 = layout_optimiser(iter,indiv,dim,pc,pm,shots,perc,e,backend,level)
             end = time.time()
             execution_1.append(exe_result_1[0])
             time1.append(end-start)
             iter_exe = np.vstack((iter_exe,np.array(exe_result_1[1])))
             iter_worst = np.vstack((iter_worst,np.array(exe_result_1[3])))
             iter_median = np.vstack((iter_median,np.array(exe_result_1[4])))
             best_indiv_mat1 = np.vstack((best_indiv_mat1,np.array(exe_result_1[2])))             
             # plot_results(exe_result_1[1],e) # visualise the fitness plot
       
        # Uncomment to display on the terminal.     
        # print("The depth obtained in all the executions: ", execution_1)
        # print("The best depth obtained is: ", np.amin(np.array(execution_1)))
        # print("The worst depth obtained is: ", np.amax(np.array(execution_1)))
        # write down the results in an excel
        book = xlwt.Workbook()
        sheet = book.add_sheet("our_routine")
        row_stats = 0
        row_exe = 6
        row_iter = 10
        row_layout_config = row_iter + iter + 5 # just enough margin to separate them
        sheet.write(row_stats,0,"best")
        sheet.write(row_stats,1,"worst")
        sheet.write(row_stats,2,"Median")
        sheet.write(row_stats,3,"std")
        sheet.write(row_stats,4,"Median Time in Seconds")
        sheet.write(row_exe,0,"Best Fitness in each Execution (col ID = execution ID)")      
        sheet.write(row_iter,0,"Best Fitness obtained ine each iteration of each execution (col ID = execution ID)")      
        sheet.write(row_iter,exe+5,"Worst Fitness obtained ine each iteration of each execution (col ID = execution ID)")
        sheet.write(row_iter,((2*exe)+10),"Meadian Fitness Population obtained ine each iteration of each execution (col ID = execution ID)")        
        sheet.write(row_layout_config,0,"Best Layout Configurations found in each execution (col ID = execution ID)")      
        sheet.write(row_stats+1,0,float(np.amin(np.array(execution_1))))
        sheet.write(row_stats+1,1,float(np.amax(np.array(execution_1))))
        sheet.write(row_stats+1,2,statistics.median(execution_1))
        sheet.write(row_stats+1,4,statistics.median(time1))
        if exe > 1:
            sheet.write(row_stats+1,3,statistics.stdev(execution_1))
        iter_exe = np.delete(iter_exe,0,0) # delete the first row of unusueful Os
        iter_worst = np.delete(iter_worst,0,0) # delete the first row of unusueful Os
        iter_median = np.delete(iter_median,0,0) # delete the first row of unusueful Os
        best_indiv_mat1 = np.delete(best_indiv_mat1,0,0) # delete the first row of unusueful Os
        # write down the best fitness obtained in each iteration
        for index3 in range(exe):
            sheet.write(row_exe+1,index3,execution_1[index3])
            for index4 in range(iter):
                sheet.write(row_iter+1+index4,index3,iter_exe[index3,index4])
                sheet.write(row_iter+1+index4,(exe+5+index3),iter_worst[index3,index4])
                sheet.write(row_iter+1+index4,(((2*exe)+10)+index3),iter_median[index3,index4])
        # write down the best configurations
        for indxx in range(exe):
            for indxx2 in range(dim):
                sheet.write(row_layout_config+1+indxx2,indxx,float(best_indiv_mat1[indxx,indxx2]))
        """
        Execute the IBM Optimiser
        """
        print("  Step 2 using the IBMQ Optimisation Routine ")
        execution_2 = []; # stock the depth of the circuits obtained IBMQ routine
        time2 = [] # stores the execution time of all the executions
        best_indiv_mat2 = np.array([0]*dim); # store the best individuals in each execution
        for e in range(exe):
             start = time.time()
             exe_result_2 = fitness2(level,shots,backend,qubits_num)
             end = time.time()
             execution_2.append(exe_result_2[0])
             best_indiv_mat2 = np.vstack((best_indiv_mat2,np.array(exe_result_2[1])))
             time2.append(end-start)
        # print("The depth obtained in all the executions: ", execution_2)
        # print("The best depth obtained is: ", np.amin(np.array(execution_2)))
        # print("The worst depth obtained is: ", np.amax(np.array(execution_2)))
        sheet_ibm = book.add_sheet("IBM_routine")
        sheet_ibm.write(row_stats,0,"best")
        sheet_ibm.write(row_stats,1,"worst")
        sheet_ibm.write(row_stats,2,"Median")
        sheet_ibm.write(row_stats,3,"std")
        sheet_ibm.write(row_stats,4,"Median Time in Seconds")
        sheet_ibm.write(row_exe,0,"Best Fitness in each Execution (col ID = execution ID)") 
        sheet_ibm.write(row_layout_config,0,"Best Layout Configurations found in each execution (col ID = execution ID)")      
        sheet_ibm.write(row_stats+1,0,float(np.amin(np.array(execution_2))))
        sheet_ibm.write(row_stats+1,1,float(np.amax(np.array(execution_2))))
        sheet_ibm.write(row_stats+1,2,statistics.median(execution_2))   
        sheet_ibm.write(row_stats+1,4,statistics.median(time2))   
        if exe > 1:
            sheet_ibm.write(row_stats+1,3,statistics.stdev(execution_2))
        best_indiv_mat2 = np.delete(best_indiv_mat2,0,0) # delete the first row of unusueful Os
        for index5 in range(exe):
            sheet_ibm.write(row_exe+1,index5,execution_2[index5])
        # write down the best configurations
        for indxx in range(exe):
            for indxx2 in range(dim):
                sheet_ibm.write(row_layout_config+1+indxx2,indxx,float(best_indiv_mat2[indxx,indxx2]))
        # in case run locally uncomment    
        # name_book = "results_"+machine+"_"+str(qubits_num)+"_qubits"+"("+str(e)+").xls" 
        # in case run on picasso uncomment    
        name_book = "results_"+machine+"_"+str(qubits_num)+"_qubits_("+str(sys.argv[1])+").xls"
        book.save(name_book) 
    return;
   
    
if __name__ == "__main__":
    main()
            
