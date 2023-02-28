#!/bin/bash
folder=`pwd`
echo $folder
for i in {1..30}
do
cd ${folder}/exec_65qubits_${i}/exec
mv 'results_FakeBrooklyn_65_qubits_('${i}').xls' ${folder}
mv 'results_FakeManhattan_65_qubits_('${i}').xls' ${folder}
cd ${folder} 
rm -r exec_65qubits_${i} 
done

