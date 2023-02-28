#!/bin/bash
folder=`pwd`
echo $folder
for i in {1..30}
do
cd ${folder}/exec_16qubits_${i}/exec
mv 'results_FakeGuadalupe_16_qubits_('${i}').xls' ${folder}
mv 'results_FakeLegacyRueschlikon_16_qubits_('${i}').xls' ${folder}
mv 'results_FakeRueschlikon_16_qubits_('${i}').xls' ${folder}
cd ${folder} 
rm -r exec_16qubits_${i} 
done

