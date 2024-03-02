@Buntender and I are in the process of reimplementing ICT[https://arxiv.org/abs/2310.14189v1] on CIFAR10 using PyTorch.

|  epoch   | initial_timesteps  | final_timesteps | ema | FID |
|  ----    |    ----   |    ----    |     ----   |   ----  | 
|   2048   |     10    |    1280    |    0.9999  | 14.7302532   |
|   200    |     10    |    150     |    0.999   |         |
|   200    |     10    |    1280    |    0.999   | 361.96563 |
|   200    |     10    |    150     |    0.99    | 54.1811   |
|   200    |     2     |    150     |    0.99    |  40.65    |
