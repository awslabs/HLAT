## Getting Started

1. Download the repo to shared disk.
2. Use image `156623026445.dkr.ecr.us-west-2.amazonaws.com/neuronx_distributed:fanhaozh_2.16.1_fsx_timeoutpatch`
3. Start the job with compilation:
```
command: ["bash"]
args: ["/PATH_TO_llm-pretrain/llama2_70b/compile_and_run.sh", "compile"]
```
4. Run the job:
```
command: ["bash"]
args: ["/PATH_TO_llm-pretrain/llama2_70b/compile_and_run.sh", "run"]
```
5. You will see the outputs
```
LOG Wed Mar 20 05:40:20 2024 - (0, 1) step_loss : 12.0000 learning_rate : 7.50e-08 throughput : 5.33 param_norm : 4576.0 grad-norm : 13.980448722839355
LOG Wed Mar 20 05:41:09 2024 - (0, 2) step_loss : 12.0000 learning_rate : 1.50e-07 throughput : 8.47 param_norm : 4576.0 grad-norm : 14.002593994140625
LOG Wed Mar 20 05:41:41 2024 - (0, 3) step_loss : 12.0000 learning_rate : 2.25e-07 throughput : 11.22 param_norm : 4576.0 grad-norm : 14.01562213897705
LOG Wed Mar 20 05:42:15 2024 - (0, 4) step_loss : 12.0000 learning_rate : 3.00e-07 throughput : 13.33 param_norm : 4576.0 grad-norm : 13.960996627807617
LOG Wed Mar 20 05:42:48 2024 - (0, 5) step_loss : 12.0000 learning_rate : 3.75e-07 throughput : 15.04 param_norm : 4576.0 grad-norm : 13.956029891967773
LOG Wed Mar 20 05:43:21 2024 - (0, 6) step_loss : 12.0000 learning_rate : 4.50e-07 throughput : 16.44 param_norm : 4576.0 grad-norm : 14.055672645568848
LOG Wed Mar 20 05:43:54 2024 - (0, 7) step_loss : 12.0000 learning_rate : 5.25e-07 throughput : 17.63 param_norm : 4576.0 grad-norm : 14.08835220336914
LOG Wed Mar 20 05:44:27 2024 - (0, 8) step_loss : 12.0000 learning_rate : 6.00e-07 throughput : 18.65 param_norm : 4576.0 grad-norm : 14.002195358276367
LOG Wed Mar 20 05:45:00 2024 - (0, 9) step_loss : 11.9375 learning_rate : 6.75e-07 throughput : 19.51 param_norm : 4576.0 grad-norm : 14.133256912231445
LOG Wed Mar 20 05:45:33 2024 - (0, 10) step_loss : 11.9375 learning_rate : 7.50e-07 throughput : 20.27 param_norm : 4576.0 grad-norm : 14.229729652404785
LOG Wed Mar 20 05:46:07 2024 - (0, 11) step_loss : 11.9375 learning_rate : 8.25e-07 throughput : 29.52 param_norm : 4576.0 grad-norm : 14.395245552062988
LOG Wed Mar 20 05:46:39 2024 - (0, 12) step_loss : 11.9375 learning_rate : 9.00e-07 throughput : 31.05 param_norm : 4576.0 grad-norm : 15.283899307250977
```
