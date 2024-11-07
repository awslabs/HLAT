set -x

# check
echo "pip list | grep neuron"
pip list | grep neuron
echo "apt list | grep neuron"
apt list | grep neuron

# check zombie
lsof -i:44000
lsof -i:23456

# check neuron device
neuron-ls
