#!/bin/bash
folder=`pwd`
echo $folder
for i in {1..30}
do
cd ${folder}/exec_27qubits_${i}/exec
mv 'results_FakeMumbai_27_qubits_('${i}').xls' ${folder}
mv 'results_FakeParis_27_qubits_('${i}').xls' ${folder}
mv 'results_FakeSydney_27_qubits_('${i}').xls' ${folder}
mv 'results_FakeToronto_27_qubits_('${i}').xls' ${folder}
cd ${folder} 
rm -r exec_27qubits_${i} 
done

