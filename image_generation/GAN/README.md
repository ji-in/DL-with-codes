# Generative Adversarial Networks (GAN) [NIPS 2014] 구현하기

[논문 링크](https://arxiv.org/abs/1406.2661)

[논문 간단 정리](https://github.com/ji-in/DL-with-codes/blob/main/image_generation/GAN/sum_up.md)

[구현 일지](https://github.com/ji-in/DL-with-codes/blob/main/image_generation/GAN/record.md)

## Configuration

Ubuntu20.04

Python 

PyTorch

## Usage

### ArgumentParser

`--gpu_id` : cpu는 -1, gpu는 0

`--batch_size` : 미니 배치 크기

`--num_epochs` : epoch 몇 번 돌릴 것인가

`--mode` : default는 train이다. test 돌리고 싶으면 test라고 명시해줘야 한다.

### Train

```
$ python gan.py --gpu_id 0
```

### Test

```
$ python gan.py --mode test --gpu_id 0
```

