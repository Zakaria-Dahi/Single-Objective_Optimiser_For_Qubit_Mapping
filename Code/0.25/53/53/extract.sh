#!/bin/bash
folder=`pwd`
echo $folder
for i in {1..30}
do
cd ${folder}/exec_53qubits_${i}/exec
mv 'results_FakeLegacyRochester_53_qubits_('${i}').xls' ${folder}
mv 'results_FakeRochester_53_qubits_('${i}').xls' ${folder}
cd ${folder} 
rm -r exec_53qubits_${i} 
done

