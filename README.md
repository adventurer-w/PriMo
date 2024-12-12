# Physical Privacy Protection for Human Action Recognition: A Portable, Scalable, and Secure Approach

## Introduction

The widespread adoption of smart devices and surveillance systems has brought substantial benefits to public safety, smart homes, and health monitoring, while elevating privacy protection to a crucial concern, particularly for human action recognition technologies. Traditional video encryption methods rely on computational algorithms, which fail to fully mitigate privacy risks inherent during video capture. To tackle this issue, we propose a novel physical-layer privacy protection approach called Lens Privacy Sealing, offering an intuitive alternative to algorithmic encryption through simple hardware modifications of existing devices, enabling adjustable privacy levels to meet different recognition requirements. Conventional HAR techniques often follow a two-stage process, where human detection in encrypted videos becomes challenging due to degraded visual information, leading to significant performance loss. To balance privacy and performance, we introduce a single-stage recognition framework using a language-image pre-training model and novel transformers for spatio-temporal integration. Additionally, we develop modules to minimize interference from privacy masks, ensuring effective motion capture while preserving privacy. Extensive experiments validate that our proposed method achieves effective privacy protection while maintaining competitive recognition accuracy.

![pipeline](assets/pipeline.png)

## Installation

To set up the environment, follow the steps below:

```bash
conda create -n PriMo python=3.7
conda activate PriMo
pip install -r requirements.txt
```

To install Apex, use the following commands:

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

We provide the complete environment configuration in `requirements.yml` for your reference.

---

## Datasets

Please refer to our [repository](https://github.com/adventurer-w/NTU-Encrypted) for instructions on downloading and preprocessing the NTU-Encrypted dataset.

---

## Training

To train the PriMo model on the NTU-Encrypted dataset using 4 GPUs, execute the following command:

```bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port=25658 ntu120_main.py \
    --config configs/NTU/NTU120_XSet.yaml \
    --distributed True \
    --accumulation-steps 2 \
    --output output/ntu_encrypted
```

### Notes:

- If your system has limited GPU memory or fewer GPUs, you can adjust the `--accumulation-steps` parameter to maintain the overall batch size.
- Configuration files are located in the `configs` directory. Ensure that the correct dataset path is specified in the configuration.

### Pretrained CLIP Model:

The pretrained CLIP model will be automatically downloaded. Alternatively, you can manually specify the path using the following option:

```bash
--pretrained /PATH/TO/PRETRAINED
```

---

## Testing

To test the PriMo model on the NTU-Encrypted dataset using 4 GPUs, execute the following command:

```bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port=25658 ntu120_main.py \
    --config configs/NTU/NTU120_XSet.yaml \
    --resume /PATH/TO/CKPT \
    --output output/ntu_encrypted \
    --only_test True
```
