timestamp=$(date +%s)

mode="train"
model='SGPR'
batch_size=1024
appliance='Refrigerator'
device="cuda"
lr=0.1
epochs=100
seed=0
extra="None"
n_cpus=1
n_restarts=3
mkdir -p logs/${mode}_${appliance}_${model}_${batch_size}_${device}_${lr}_${epochs}_${seed}_${extra}_${n_cpus}_${n_restarts}
python -u new_files/run.py --mode ${mode} --appliance ${appliance} --model ${model} --batch_size ${batch_size} --device ${device} --lr ${lr} --epochs ${epochs} --seed ${seed} --extra ${extra} --n_cpus ${n_cpus} --n_restarts ${n_restarts} > logs/${mode}_${appliance}_${model}_${batch_size}_${device}_${lr}_${epochs}_${seed}_${extra}_${n_cpus}_${n_restarts}/stdout.${timestamp}.log

# mode="test"
# model='SGPR'
# batch_size=1024
# appliance='Refrigerator'
# device="cuda"
# lr=0.01
# epochs=100
# seed=0
# extra="None"
# n_cpus=1
# n_restarts=3
# mkdir -p logs/${mode}_${appliance}_${model}_${batch_size}_${device}_${lr}_${epochs}_${seed}_${extra}_${n_cpus}_${n_restarts}
# python -u new_files/run.py --mode ${mode} --appliance ${appliance} --model ${model} --batch_size ${batch_size} --device ${device} --lr ${lr} --epochs ${epochs} --seed ${seed} --extra ${extra} --n_cpus ${n_cpus} --n_restarts ${n_restarts} > logs/${mode}_${appliance}_${model}_${batch_size}_${device}_${lr}_${epochs}_${seed}_${extra}_${n_cpus}_${n_restarts}/stdout.${timestamp}.log