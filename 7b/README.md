## Getting Started

1. Download the repo to shared disk.
2. Use image `448005706873.dkr.ecr.us-west-2.amazonaws.com/neuronx_distributed:nxd-2.15.2all`
3. Start the job with compilation:
```
command: ["bash"]
args: ["/PATH_TO_AireScaleNxDModels/llama2/compile_and_run.sh", "compile"]
```
4. Run the job:
```
command: ["bash"]
args: ["/PATH_TO_AireScaleNxDModels/llama2/compile_and_run.sh", "run"]
```
5. You will see the outputs 
```
LOG Wed Dec  6 21:16:42 2023 - (0, 1) step_loss : 13.1875 learning_rate : 3.00e-07 throughput : 1.38 param_norm : 1536.0 grad-norm : 8.1875
LOG Wed Dec  6 21:17:13 2023 - (0, 2) step_loss : 10.5000 learning_rate : 4.50e-07 throughput : 2.64 param_norm : 1536.0 grad-norm : 8.125
LOG Wed Dec  6 21:17:21 2023 - (0, 3) step_loss : 13.1250 learning_rate : 6.00e-07 throughput : 3.92 param_norm : 1544.0 grad-norm : 8.125
LOG Wed Dec  6 21:17:24 2023 - (0, 4) step_loss : 13.1250 learning_rate : 7.50e-07 throughput : 5.21 param_norm : 1536.0 grad-norm : 8.5
```



