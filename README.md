# amx-basic-example

This repository contains basic example code using Intel AMX.

# Matrix product operation using AMX

Pay attention to the memory layout of B in the multiplication instruction A * B.

- `int8_mut`: int8 matrix product operation
```
cd int8_mul
icc main.c -o int8_mul
```

- `bf16_mul`: BF16 matrix product operation
```
cd bf16_mul
icc main.c -o bf16_mul
```

# Convolutional operation using AMX

Convolutional operations using AMX require unique handling.
The approach is similar to im2col, which is commonly used on GPUs.
Note that AMX multiply instructions are accumulative, not overwritten.

- `int8_conv`: int8 convolution operation
```
cd int8_conv
icc int8_conv -o int8_conv
```
