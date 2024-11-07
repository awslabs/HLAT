# HLAT

Open source code for paper (HLAT: High-quality Large Language Model Pre-trained on AWS Trainium: https://arxiv.org/abs/2404.10630) and blog (https://aws.amazon.com/blogs/machine-learning/end-to-end-llm-training-on-instance-clusters-with-over-100-nodes-using-aws-trainium).

## Get Started

Steps to reproduce the results:

1. Prepare a slurm cluster that has trn1.32xlarge instances.
2. Get access and download Llama tokenizer from https://huggingface.co/meta-llama/Llama-2-7b.
3. Download the repo to shared disk.
4. Prepare dataset, any dataset in Apache Arrow format and each row contains the field "text".
5. Change tokenizer path and dataset path:
   - 7b: `7b/tp_zero1_llama2_7b_hf_pretrain.sh` line 67 & 194 & 196.
   - 70b: `70b/tp_zero1_llama2_70b_hf_pretrain.sh` line 103 & 187.
6. Also change the tensorboard path at the end of training script.
7. Compile and run the job:
   - compile: `sbatch run_llama.slurm` in folder `7b` or `70b`
   - run: change "compile" to "run" in `run_llama.slurm`, then `sbatch run_llama.slurm`

Tips:

- Sometimes installation script fails, just retry.
- Make sure output path have enough space to store checkpoints.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

## Reference

If you found HLAT useful in your research or applications, please cite using the following BibTeX:
```
@software{HLAT,
  title = {HLAT: High-quality Large Language Model Pre-trained on AWS Trainium},
  author = {Haozheng Fan, Hao Zhou, Guangtai Huang, Parameswaran Raman, Xinwei Fu, Gaurav Gupta, Dhananjay Ram, Yida Wang, Jun Huan},
  url = {https://github.com/awslabs/HLAT/},
  year={2024}
}
```
```
@article{HLAT,
  title={HLAT: High-quality Large Language Model Pre-trained on AWS Trainium},
  author={Haozheng Fan, Hao Zhou, Guangtai Huang, Parameswaran Raman, Xinwei Fu, Gaurav Gupta, Dhananjay Ram, Yida Wang, Jun Huan},
  journal={arXiv preprint arXiv:2404.10630},
  year={2024}
}
```
