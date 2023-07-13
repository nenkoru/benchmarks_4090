# Benchmarks of a RTX 4090 of mine running within a QEMU/KVM in Proxmox

## Environment:

All the tests were run within a virtual machine(qemu) which is run using proxmox 7.4-3.
<image of proxmox hardware here>
```
CPU: Intel Xeon 2696v4 2.2Ghz (16 vCPUs)
Storage: Samsung SSD 870 EVO
OS: Linux gpu-node 5.15.0-76-generic #83-Ubuntu SMP Thu Jun 15 19:16:32 UTC 2023 x86_64 x86_64 x86_64 GNU/LinuxDISTRIB_DESCRIPTION="Ubuntu 22.04.2 LTS"
Python 3.10.12 (main, Jul  5 2023, 18:54:27) [GCC 11.2.0] on linux
Torch version: Version: 2.1.0.dev20230709+cu121
import torch;torch.version.cuda -> '12.1'
Cuda compilation tools, release 12.1, V12.1.105
Build cuda_12.1.r12.1/compiler.32688072_0
```


## Benchmarks used:
```
https://github.com/huggingface/pytorch-image-models/tree/394e8145551191ae60f672556936314a20232a35
https://github.com/pytorch/examples/tree/7f7c222b355abd19ba03a7d4ba90f1092973cdbc
```

[Reddit post acompanying this repo](https://www.reddit.com/r/MachineLearning/comments/14yw72v/r_nvidia_rtx_4090_ml_benchmarks_under_qemukvm/)
