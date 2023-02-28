#!/bin/bash
folder=`pwd`
echo $folder
for i in {1..30}
do
cd ${folder}/exec_20qubits_${i}/exec
mv 'results_FakeLegacySingapore_20_qubits_('${i}').xls' ${folder}
mv 'results_FakeLegacyTokyo_20_qubits_('${i}').xls' ${folder}
mv 'results_FakePoughkeepsie_20_qubits_('${i}').xls' ${folder}
mv 'results_FakeSingapore_20_qubits_('${i}').xls' ${folder}
cd ${folder} 
rm -r exec_20qubits_${i} 
done

