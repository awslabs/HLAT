#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set -x

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SCRIPT_DIR_NAME=$(basename $SCRIPT_DIR)
USAGE="Usage: $0 VENV_PATH"

if [ $# -lt 1 ] || [ $# -gt 1 ]; then
    echo $USAGE
    exit 1
fi

VENV_PATH=$1

echo "VENV_PATH = $VENV_PATH"

#############################################
# install runtime

echo "Neuron SDK Release 2.20.0"
# Update OS packages
sudo apt-get update -y
# Configure Linux for Neuron repository updates
. /etc/os-release
sudo tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF
deb https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main
EOF
wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo apt-key add -

sudo dpkg --configure -a

# Remove preinstalled packages and Install Neuron Driver and Runtime
sudo apt-get remove aws-neuron-dkms -y
sudo apt-get remove aws-neuronx-dkms -y
sudo apt-get remove aws-neuronx-oci-hook -y
sudo apt-get remove aws-neuronx-runtime-lib -y
sudo apt-get remove aws-neuronx-collectives -y
sudo apt-get install aws-neuronx-dkms=2.18.12.0 -y
sudo apt-get install aws-neuronx-oci-hook=2.5.3.0 -y
sudo apt-get install aws-neuronx-runtime-lib=2.22.14.0* -y
sudo apt-get install aws-neuronx-collectives=2.22.26.0* -y

# Remove pre-installed package and Install Neuron Tools
sudo apt-get remove aws-neuron-tools  -y
sudo apt-get remove aws-neuronx-tools  -y
sudo apt-get install aws-neuronx-tools=2.19.0.0 -y

# Install EFA Driver (only required for multi-instance training)
cd /tmp
curl -O https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz
wget https://efa-installer.amazonaws.com/aws-efa-installer.key && gpg --import aws-efa-installer.key
cat aws-efa-installer.key | gpg --fingerprint
wget https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz.sig && gpg --verify ./aws-efa-installer-latest.tar.gz.sig
tar -xvf aws-efa-installer-latest.tar.gz
cd aws-efa-installer && sudo bash efa_installer.sh --yes
cd ..
sudo rm -rf aws-efa-installer-latest.tar.gz aws-efa-installer
cd $SCRIPT_DIR

#############################################
# install venv
python3 -m venv $VENV_PATH
source $VENV_PATH/bin/activate
pip install -U pip

# Install packages from repos
python -m pip config set global.extra-index-url "https://pip.repos.neuron.amazonaws.com"
python -m pip install torch-neuronx=="1.13.1.1.16.0" neuronx-cc=="2.15.128.0" neuronx_distributed=="0.9.0" torchvision
pip3 install transformers==4.31.0
pip3 install tensorboard
pip3 install datasets
pip3 install omegaconf
pip3 install regex sentencepiece
pip3 install pytorch-lightning==2.1.0

chown ubuntu:ubuntu -R $VENV_PATH

#############################################
# # backup
PKG_PATH="$VENV_PATH/lib/python3.10/site-packages"

echo "PKG_PATH = $VENV_PATH"

# cp $PKG_PATH/neuronx_distributed/trainer/checkpoint.py $PKG_PATH/neuronx_distributed/trainer/checkpoint.py.old
cp $PKG_PATH/neuronx_distributed/pipeline/model.py $PKG_PATH/neuronx_distributed/pipeline/model.py.old
cp $PKG_PATH/neuronx_distributed/parallel_layers/parallel_state.py $PKG_PATH/neuronx_distributed/parallel_layers/parallel_state.py.old
cp $PKG_PATH/torch_neuronx/distributed/xrt_init.py $PKG_PATH/torch_neuronx/distributed/xrt_init.py.old
cp $PKG_PATH/torch/distributed/elastic/agent/server/api.py $PKG_PATH/torch/distributed/elastic/agent/server/api.py.old

#############################################
# # patch
# cp $SCRIPT_DIR/nxd_patches/checkpoint.py $PKG_PATH/neuronx_distributed/trainer/checkpoint.py
cp $SCRIPT_DIR/nxd_patches/model.py $PKG_PATH/neuronx_distributed/pipeline/model.py
cp $SCRIPT_DIR/nxd_patches/parallel_state.py $PKG_PATH/neuronx_distributed/parallel_layers/parallel_state.py
cp $SCRIPT_DIR/torch_neuronx_patches/xrt_init.py $PKG_PATH/torch_neuronx/distributed/xrt_init.py
cp $SCRIPT_DIR/torch_patches/api.py $PKG_PATH/torch/distributed/elastic/agent/server/api.py
