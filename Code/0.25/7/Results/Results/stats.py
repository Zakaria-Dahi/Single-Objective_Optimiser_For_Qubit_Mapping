#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 16:47:56 2022

@author: user
"""

from qiskit import transpile
from qiskit.test.mock import FakeYorktown,FakeVigo,FakeValencia,FakeToronto,FakeSydney,FakeSingapore,FakeSantiago,FakeRome,FakeRochester,FakeQuito,FakePoughkeepsie,FakeParis,FakeOurense,FakeOpenPulse3Q,FakeMumbaiV2,FakeMumbai,FakeMontreal,FakeManila,FakeManhattan,FakeLondon,FakeLima,FakeLegacyYorktown,FakeLegacyVigo,FakeLegacyValencia,FakeLegacyToronto,FakeLegacyTokyo,FakeLegacyTenerife,FakeLegacySydney,FakeLegacySingapore,FakeLegacySantiago,FakeLegacyRueschlikon,FakeLegacyRome,FakeLegacyRochester,FakeLegacyQuito,FakePoughkeepsie,FakeLegacyParis,FakeLegacyOurense,FakeLegacyMumbai,FakeLegacyMontreal,FakeLegacyMelbourne,FakeLegacyManhattan,FakeLegacyLondon,FakeLegacyLima,FakeLegacyJohannesburg,FakeLegacyEssex,FakeLegacyCasablanca,FakeLegacyCambridgeAlternativeBasis,FakeLegacyCambridge,FakeLegacyBurlington,FakeLegacyBogota,FakeLegacyBoeblingen,FakeLegacyBelem,FakeLegacyAthens,FakeLegacyAlmaden,FakeLagos,FakeJohannesburg,FakeJakarta,FakeGuadalupe,FakeEssex,FakeCasablanca,FakeCambridgeAlternativeBasis,FakeCambridge,FakeBurlington,FakeBrooklyn,FakeBogota,FakeBoeblingen,FakeBelem,FakeAthens,FakeAlmaden,FakeOpenPulse2Q, FakeTenerife, FakeMelbourne, FakeRueschlikon, FakePoughkeepsie, FakeTokyo
import numpy as np
import statistics
import xlwt
import xlrd

def test_backend(machine):
    if machine == "FakeOpenPulse2Q":        
        backend = FakeOpenPulse2Q()  
    if machine == "FakeTenerife":
        backend = FakeTenerife()
    if machine == "FakeMelbourne":
        backend = FakeMelbourne()
    if machine == "FakeRueschlikon":
        backend = FakeRueschlikon()
    if machine == "FakePoughkeepsie":
        backend = FakePoughkeepsie()
    if machine == "FakeTokyo":
        backend = FakeTokyo()
    if machine == "FakeAlmaden":
        backend = FakeAlmaden()          
    if machine == "FakeAthens":
        backend = FakeAthens()  
    if machine == "FakeBelem":
        backend = FakeBelem()  
    if machine == "FakeBoeblingen":
        backend = FakeBoeblingen()
    if machine == "FakeBogota":
        backend = FakeBogota()
    if machine == "FakeBrooklyn":
        backend = FakeBrooklyn()  
    if machine == "FakeBurlington":
        backend = FakeBurlington()
    if machine == "FakeCambridge":
        backend = FakeCambridge()
    if machine == "FakeCambridgeAlternativeBasis":
        backend = FakeCambridgeAlternativeBasis()  
    if machine == "FakeCasablanca":
        backend = FakeCasablanca()
    if machine == "FakeEssex":
        backend = FakeEssex()
    if machine == "FakeGuadalupe":
        backend = FakeGuadalupe()  
    if machine == "FakeJakarta":
        backend = FakeJakarta()
    if machine == "FakeJohannesburg":
        backend = FakeJohannesburg()
    if machine == "FakeLagos":
        backend = FakeLagos()  
    if machine == "FakeLegacyAlmaden":
        backend = FakeLegacyAlmaden()
    if machine == "FakeLegacyAthens":
        backend = FakeLegacyAthens()
    if machine == "FakeLegacyBelem":
        backend = FakeLegacyBelem()  
    if machine == "FakeLegacyBoeblingen":
        backend = FakeLegacyBoeblingen()
    if machine == "FakeLegacyBogota":
        backend = FakeLegacyBogota()            
    if machine == "FakeLegacyBurlington":
        backend = FakeLegacyBurlington()
    if machine == "FakeLegacyCambridge":
        backend = FakeLegacyCambridge()  
    if machine == "FakeLegacyCambridgeAlternativeBasis":
        backend = FakeLegacyCambridgeAlternativeBasis()
    if machine == "FakeLegacyCasablanca":
        backend = FakeLegacyCasablanca()
    if machine == "FakeLegacyEssex":
        backend = FakeLegacyEssex()  
    if machine == "FakeLegacyJohannesburg":
        backend = FakeLegacyJohannesburg()
    if machine == "FakeLegacyLima":
        backend = FakeLegacyLima()
    if machine == "FakeLegacyLondon":
        backend = FakeLegacyLondon()  
    if machine == "FakeLegacyManhattan":
        backend = FakeLegacyManhattan()
    if machine == "FakeLegacyMelbourne":
        backend = FakeLegacyMelbourne() 
    if machine == "FakeLegacyMontreal":
        backend = FakeLegacyMontreal()
    if machine == "FakeLegacyMumbai":
        backend = FakeLegacyMumbai()
    if machine == "FakeLegacyOurense":
        backend = FakeLegacyOurense()  
    if machine == "FakeLegacyParis":
        backend = FakeLegacyParis()
    if machine == "FakePoughkeepsie":
        backend = FakePoughkeepsie() 
    if machine == "FakeLegacyQuito":
        backend = FakeLegacyQuito()
    if machine == "FakeLegacyRochester":
        backend = FakeLegacyRochester()
    if machine == "FakeLegacyRome":
        backend = FakeLegacyRome()  
    if machine == "FakeLegacyRueschlikon":
        backend = FakeLegacyRueschlikon()
    if machine == "FakeLegacySantiago":
        backend = FakeLegacySantiago() 
    if machine == "FakeLegacySingapore":
        backend = FakeLegacySingapore()  
    if machine == "FakeLegacySydney":
        backend = FakeLegacySydney()
    if machine == "FakeLegacyTenerife":
        backend = FakeLegacyTenerife() 
    if machine == "FakeLegacyTokyo":
        backend = FakeLegacyTokyo()
    if machine == "FakeLegacyToronto":
        backend = FakeLegacyToronto()
    if machine == "FakeLegacyValencia":
        backend = FakeLegacyValencia()  
    if machine == "FakeLegacyVigo":
        backend = FakeLegacyVigo()
    if machine == "FakeLegacyYorktown":
        backend = FakeLegacyYorktown() 
    if machine == "FakeLima":
        backend = FakeLima()
    if machine == "FakeLondon":
        backend = FakeLondon()
    if machine == "FakeManhattan":
        backend = FakeManhattan()  
    if machine == "FakeManila":
        backend = FakeManila()
    if machine == "FakeMontreal":
        backend = FakeMontreal() 
    if machine == "FakeMumbai":
        backend = FakeMumbai()
    if machine == "FakeMumbaiV2":
        backend = FakeMumbaiV2()
    if machine == "FakeOpenPulse3Q":
        backend = FakeOpenPulse3Q()  
    if machine == "FakeOurense":
        backend = FakeOurense()
    if machine == "FakeParis":
        backend = FakeParis()             
    if machine == "FakePoughkeepsie":
        backend = FakePoughkeepsie() 
    if machine == "FakeQuito":
        backend = FakeQuito()
    if machine == "FakeRochester":
        backend = FakeRochester()
    if machine == "FakeRome":
        backend = FakeRome()  
    if machine == "FakeSantiago":
        backend = FakeSantiago()
    if machine == "FakeSingapore":
        backend = FakeSingapore()              
    if machine == "FakeSydney":
        backend = FakeSydney() 
    if machine == "FakeToronto":
        backend = FakeToronto()
    if machine == "FakeValencia":
        backend = FakeValencia()
    if machine == "FakeVigo":
        backend = FakeVigo()  
    if machine == "FakeYorktown":
        backend = FakeYorktown()    
    return backend;


def main():
    backend_lists = ['FakeLegacyRueschlikon','FakeGuadalupe','FakeRueschlikon']
    configs = []
    for machine in backend_lists:
        backend = test_backend(machine)
        # recover the number of qubits in the machine
        qubits_num = backend.configuration().n_qubits;    
        configs.append(qubits_num);
    # extract the classes of machines and the number of qubits they have
    print(configs)
    configs = np.copy(configs[:][:])
    configs_qubits = np.unique(configs)
    configs_qubits_sorted = np.sort(configs_qubits)
    # browse results from smallest number of qubits to the largest
    book = xlwt.Workbook()
    for i in range(len(configs_qubits_sorted)):
        sheet = book.add_sheet(str(configs_qubits_sorted[i])+"_qubits")
        index = 0
        for machine in backend_lists:
            backend = test_backend(machine)
            num_qubits_processed = backend.configuration().n_qubits
            if configs_qubits_sorted[i] == num_qubits_processed:
               sheet.write(index,0,"results using the machine "+ machine)
               sheet.write(index+1,1,"Best")
               sheet.write(index+1,2,"Worst")
               sheet.write(index+1,3,"Median")
               sheet.write(index+1,4,"STD")
               sheet.write(index+1,5,"Time")
               sheet.write(index+2,0,"Us")
               sheet.write(index+3,0,"IBMQ")
               # browse the results
               data0 = []
               data1 = []
               data0_time = []
               data1_time = []               
               for j in range(1,31):
                   book_read = xlrd.open_workbook("results_"+machine+"_"+str(num_qubits_processed)+"_qubits_("+str(j)+").xls")
                   sheet_read0 = book_read.sheet_by_index(0)
                   data0.append(sheet_read0.cell_value(1,2))
                   data0_time.append(sheet_read0.cell_value(1,4))
                   sheet_read1 = book_read.sheet_by_index(1)
                   data1.append(sheet_read1.cell_value(1,2))
                   data1_time.append(sheet_read1.cell_value(1,4))
               # write down the results of our routine   
               sheet.write(index+2,1,np.amin(np.array(data0)))
               sheet.write(index+2,2,np.amax(np.array(data0)))
               sheet.write(index+2,3,statistics.median(data0))
               sheet.write(index+2,4,statistics.stdev(data0))
               sheet.write(index+2,5,statistics.median(data0_time))               
               # write down the results of IBM routin
               sheet.write(index+3,1,np.amin(np.array(data1)))
               sheet.write(index+3,2,np.amax(np.array(data1)))
               sheet.write(index+3,3,statistics.median(data1))
               sheet.write(index+3,4,statistics.stdev(data1))
               sheet.write(index+3,5,statistics.median(data1_time))               
               # write down the results of the executions
               col = 8
               for jj in range(len(data0)):
                   sheet.write(index+2,col+1,data0[jj])
                   sheet.write(index+3,col+1,data1[jj])
                   sheet.write(index+4,col+1,data0_time[jj])
                   sheet.write(index+5,col+1,data1_time[jj])
                   col = col+1
               index = index + 6 # set the line to write down the results of the next line
               
        book.save("stats.xls")
                
if __name__ == "__main__":
    main()

    