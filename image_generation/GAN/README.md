# Generative Adversarial Networks (GAN) [NIPS 2014] 간단 정리

[논문 링크](https://arxiv.org/abs/1406.2661)

## Adversarial nets

1. 먼저 입력 노이즈 p_z(z)를 만든다.
2. Generator G(z)는 노이즈 p_z(z)를 입력으로 받고 이미지를 생성한다.  
3. Discriminator D(x)는 입력 x를 입력으로 받고 x가 data에서 왔는지, g가 생성한 이미지에서 왔는지 판별한다.

D와 G는 two-player minimax game을 하는 것과 같다.

<img src=".\loss_function.PNG" style="zoom:70%;" />

Algorithm1은 식(1)을 최적화한다.

![](.\algorithm1.PNG)

 

```
for number of training iterations do
    for k steps do
        m개의 노이즈를 만든다. -> z
        데이터셋에서 m개의 이미지를 선택한다. -> x
        stochastic gradient를 ascending해서 discriminator를 업데이트한다. -> 목적 함수의 입력으로 z와 x를 사용한다.
    end for
    m개의 노이즈를 만든다 -> z
    stochastic gradient를 descending해서 generator를 업데이트 한다. -> 목적 합수의 입력으로 z를 사용한다.
end for
```

Algorithm1을 보면, discriminator를 k번 update 하고, generator를 1번 update 하는 것을 볼 수 있다.

