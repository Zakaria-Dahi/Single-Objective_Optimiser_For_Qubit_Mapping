#!/bin/bash
folder=`pwd`
echo $folder
for i in {1..30}
do
cd ${folder}/exec_7qubits_${i}/exec
mv 'results_FakeCasablanca_7_qubits_('${i}').xls' ${folder}
mv 'results_FakeJakarta_7_qubits_('${i}').xls' ${folder}
mv 'results_FakeLagos_7_qubits_('${i}').xls' ${folder}
mv 'results_FakeLegacyCasablanca_7_qubits_('${i}').xls' ${folder}
cd ${folder} 
rm -r exec_7qubits_${i} 
done

