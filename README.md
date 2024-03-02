I and @Buntender are trying to reimplement ict on cifar10 with Pytorch.

|  epoch   | initial_timesteps  | final_timesteps | ema | FID |
|  ----    |    ----   |    ----    |     ----   |   ----  | 
|   2048   |     10    |    1280    |    0.9999  | 14.7302532   |
|   200    |     10    |    150     |    0.999   |         |
|   200    |     10    |    1280    |    0.999   | 361.96563 |
|   200    |     10    |    150     |    0.99    | 54.1811   |
|   200    |     2     |    150     |    0.99    |  40.65    |
