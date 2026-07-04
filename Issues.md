
# This Project
> 首先要理解整个项目的架构，然后根据自己的需求进行修改

##  Modify architect

### Dataset
1. 将 msgpack_numpy 修改为 pickle
2. 在数据集中添加 instruction 和 rgb 数据类型

---

### Training 
- 首先，修改 Dataset ，将数据处理成模型需要的格式。

1. input
2. policy（model）
3. loss 



## Issues

> 报错之后，先自己分析一下问题，然后将自己的理解和问题让大模型分析。先小范围修改，然后提交到 Git 上，之后可以回退。

###  NCCL 超时错误（600 秒）意味着多卡训练中通信操作卡死
#### 数据长度不平衡导致某些卡处理更快/更慢
修改 collate_fn： 统一各轨迹长度（填充到 batch 内最大长度）

#### 减少 NCCL 通信次数
仅在最后一个时间步同步梯度，之前时间步使用 no_sync（DDP）的延迟同步


####  设置 NCCL 超时环境变量（可选）
```
export NCCL_TIMEOUT=1800   # 增大超时时间至 30 分钟
export NCCL_ASYNC_ERROR_HANDLING=1
```

---

### OOM
1. 首先先将 batch size 逐步下降
2. 使用 Lora 训练模型
3. 使用 deepspeed 或者 DDP

---

### 使用 DDP 问题
- module,
- 保存模型（）

---

### 模型精度太低，导致训练的 logit 变成 nan
原因是我使用 vlm.half()
这样可以



