# 概述
FourCastNet ，首个全球高分辨率人工智能天气预报模型。它是基于自适应傅里叶算子 AFNO （Adaptive Fourier Neural Operators）的模型，只不过把数据从 3 个通道的自然图像换成了 20 个通道的气象数据，也就是说，它将气象预测问题转化成图像问题。具体的原理图如下所示：

![](http://26l1b06988.qicp.vip:38000/pictures/20250607171022.png)

FourCastNet的核心架构是自适应傅里叶神经算子（Adaptive Fourier Neural Operator, AFNO），并将其整合在一个**视觉变换器（Vision Transformer, ViT）** 的骨干网络中。这个设计巧妙地结合了ViT处理长距离依赖关系的能力和AFNO高效处理高分辨率数据的能力。模型接收一个时刻的全球天气状态作为输入，输出下一个时刻（6小时后）的天气状态。

在这次实验中，我将对这一模型进行微调以使其能够分析2025年的天气情况，最终我还设计了一个MCP工具，这个工具可以用来为支持MCP协议的大语言模型提供进行气象预测的能力，让我们的微调成果有一定的实际应用价值。
# 数据集处理
## 数据集获取
我们选择ECWMF提供的ERA5数据集作为我们的数据来源，ECWMF提供了一个用于自动化下载的API，我们可以在Python中方便地使用它，`nc_download.py`提供了一个交互的数据集下载工具。
注意，由于在下载逐气压等级的气象数据时还需要指明所需的气压大小，而在下载地表数据时不需要提供，因此我们需要发送两个不同的请求，一个下载地表数据，一个下载逐气压数据。
FourCastNet需要的20个气象变量如下所示：

![](http://26l1b06988.qicp.vip:38000/pictures/20250615142140.png)

## 数据集格式转换
FourCastNet的输入数据格式为一个按照一定的顺序排列好的HDF5文件，而我们直接从ERA5数据集中下载下来的文件是NetCDF4格式的，所以我们需要用一个脚本在两者之间进行转换，转换脚本在`nc_to_h5.py`文件中，这个文件参考了NVLab官方仓库中的脚本，但是由于我们数据量较小，没必要使用MPI加速转换，因此只提供了一个单线程的版本，实践证明这个脚本能够满足我们的需求。
## 踩坑点
一开始，我嫌Kaggle从CDS上下载数据过慢，经过搜索发现GCS提供了一个类似于CDS上ERA5数据集的备份`public-data-arco-era5`，我发现从这里获取数据的速度似乎更快，于是重新写了一个脚本使用GCS上的数据，但是，我忘记了最重要的一点：测试数据集的完整性。我在使用从这里获取的数据跑了一个星期（真的整整一个星期）的微调后，效果总是不尽如人意，后来我才突然意识到这里可能存在的问题，检查了一下，天塌了：

![3d450b5fadef66d0b043b33bcf2ef373.png](http://26l1b06988.qicp.vip:38000/pictures/3d450b5fadef66d0b043b33bcf2ef373.png)

这时我才意识到，先前微调效果不足的罪魁祸首是数据集，而我在这个数据集上做了各种各样不同的参数组合的微调测试，还尝试了LoRA这样的技术，但效果都没有得到实质性的改进。

# 推理
这里的推理脚本我参考了github仓库[climatechange-ai-tutorials/fourcastnet: Learn how to use FourCastNet, a weather model based on deep learning, to obtain short to medium-range forecasts of crucial atmospheric variables such as surface wind velocities.](https://github.com/climatechange-ai-tutorials/fourcastnet)给出的示例，它使用数据集的第一个时间点作为输入，然后进行指定步数的推理，将推理结果与数据集上相应时间的真实数据进行比较，这是一个很好的用来验证模型预测天气能力的示例，接下来模型预测能力好坏的对比都将采用这个脚本中的推理函数。
# 现有模型的问题
NVidia官方提供的预训练模型同时还提供了一个有40个时间步的测试集，我们使用上面提到的推理脚本和这个测试集进行测试，可以看到，在各个变量的预测上，这个模型做的都是很不错的。

![](http://26l1b06988.qicp.vip:38000/pictures/20250615144809.png)

![](http://26l1b06988.qicp.vip:38000/pictures/20250615145533.png)

![](http://26l1b06988.qicp.vip:38000/pictures/20250615145400.png)

![](http://26l1b06988.qicp.vip:38000/pictures/20250615145434.png)

但是，当我们使用2025年的数据做预测时，它的预测误差瞬间变得特别大
![](http://26l1b06988.qicp.vip:38000/pictures/20250615145858.png)

这很可能是由于**分布漂移**导致的，论文中提到了，提供的预训练数据是由1970-2017年的数据训练得到的。而在2017-2025年的这段时间内，地球的气候发生了许多变化， 使得先前预训练的模型无法很好地适应这些变化，于是我们需要通过**迁移学习**使得模型对近期的气象环境进行适应。而最常用的**迁移学习**方法便是微调。
# 微调

微调的过程中我多次迭代了微调策略
## iter 0 - 测试可行性
首先，我们构建一个最简单的微调框架，这个框架采取全参数微调，配置如下：
``` python
class Config:
    # --- 路径配置 ---
    h5_path = '/kaggle/input/2024-dec-h5/output_h5/2024-02.h5'
    pretrained_model_path = '/kaggle/input/fourcastnet/pytorch/default/2/fcn-v0.1/fcn-v0.1/model_weights/FCN_weights_v0/backbone.ckpt'
    output_dir = '/kaggle/working/finetuned_2024_feb'
    finetuned_model_save_path = os.path.join(output_dir, 'finetuned_model_best.pt')
    STATS_DIR = os.path.join(output_dir, 'stats')

    # --- 数据配置 ---
    CHANNELS = list(range(20))
    train_days = 22
    total_days = 29
    timesteps_per_day = 4
    train_timesteps = train_days * timesteps_per_day
    total_timesteps = total_days * timesteps_per_day

    # --- 模型架构配置---
    # 此处参考了Nvidia官方仓库中的相关文件
    img_size = (720, 1440)
    patch_size = 8
    N_in_channels = 20
    N_out_channels = 20
    num_blocks = 8
    embed_dim = 768
    depth = 12
    mlp_ratio = 4.0
    drop_rate = 0.0
    drop_path_rate = 0.0
    sparsity_threshold = 0.01
    hard_thresholding_fraction = 1.0

    # --- 微调超参数 ---
    num_epochs = 5
    batch_size = 1
    learning_rate = 1e-5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 实例化配置
params = Config()
```

然后，进行基本的数据集处理
```python
print("正在加载官方提供的全局均值和标准差...")
try:
    means_full = np.load(os.path.join(params.STATS_DIR, 'global_means.npy'))
    stds_full = np.load(os.path.join(params.STATS_DIR, 'global_stds.npy'))
    means_sliced = means_full[:, params.CHANNELS, ...]
    stds_sliced = stds_full[:, params.CHANNELS, ...]
    print("官方统计数据加载并切片完成。")
except FileNotFoundError:
    print("错误：找不到官方的 global_means.npy 或 global_stds.npy。")
    raise

class HDF5MultiFileDataset(Dataset):
    def __init__(self, h5_path, time_indices, means, stds, config):
        self.h5_path = h5_path
        self.time_indices = time_indices
        self.means = torch.from_numpy(means).float().squeeze(0)
        self.stds = torch.from_numpy(stds).float().squeeze(0)
        self.config = config
        
        if not os.path.exists(self.h5_path):
            raise FileNotFoundError(f"HDF5 文件未找到: {self.h5_path}")
            
    def __len__(self):
        return len(self.time_indices) - 1

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as hf:
            input_idx = self.time_indices[idx]
            target_idx = self.time_indices[idx] + 1
            
            inp_raw = hf['fields'][input_idx, self.config.CHANNELS, :, :]
            tar_raw = hf['fields'][target_idx, self.config.CHANNELS, :, :]
            
            if inp_raw.shape[1] == 721:
                inp_raw = inp_raw[:, :-1, :]
                tar_raw = tar_raw[:, :-1, :]

            inp_raw = torch.from_numpy(inp_raw).float()
            tar_raw = torch.from_numpy(tar_raw).float()

        inp_norm = (inp_raw - self.means) / (self.stds + 1e-6)
        tar_norm = (tar_raw - self.means) / (self.stds + 1e-6)
            # 检查原始数据是否有 NaN/Inf
        if torch.isnan(inp_raw).any() or torch.isinf(inp_raw).any():
            print(f"输入数据包含 NaN 或 Inf (idx={idx}, input_idx={input_idx})")
            # 可选：替换为零或跳过该样本
            inp_raw = torch.zeros_like(inp_raw)
            abort()
    
        if torch.isnan(tar_raw).any() or torch.isinf(tar_raw).any():
            print(f"目标数据包含 NaN 或 Inf (idx={idx}, target_idx={target_idx})")
            tar_raw = torch.zeros_like(tar_raw)
            abort()
    
        # 归一化
        inp_norm = (inp_raw - self.means) / (self.stds + 1e-6)
        tar_norm = (tar_raw - self.means) / (self.stds + 1e-6)
    
        # 再次检查归一化后的数据
        if torch.isnan(inp_norm).any() or torch.isinf(inp_norm).any():
            print(f"归一化后输入数据包含 NaN 或 Inf (idx={idx})")
            inp_norm = torch.zeros_like(inp_norm)
            abort()
    
        if torch.isnan(tar_norm).any() or torch.isinf(tar_norm).any():
            print(f"归一化后目标数据包含 NaN 或 Inf (idx={idx})")
            tar_norm = torch.zeros_like(tar_norm)
            abort()
        
        return inp_norm, tar_norm

train_datasets = []
valid_datasets = []
print("正在准备数据集...")
for name, path in params.H5_PATHS.items():
    print(f"处理文件: {name} ({path})")
    try:
        with h5py.File(path, 'r') as hf:
            total_timesteps = hf['fields'].shape[0]
    except (FileNotFoundError, KeyError) as e:
        print(f"警告：无法读取文件 {path} 或其 'fields' 数据集。错误: {e}")
        continue

    train_indices = list(range(params.TRAIN_DAYS_PER_FILE * params.TIMESTEPS_PER_DAY))
    valid_indices = list(range(params.VALID_START_DAY_PER_FILE * params.TIMESTEPS_PER_DAY, total_timesteps))

    train_ds = HDF5MultiFileDataset(path, train_indices, means_sliced, stds_sliced, params)
    valid_ds = HDF5MultiFileDataset(path, valid_indices, means_sliced, stds_sliced, params)
    
    train_datasets.append(train_ds)
    valid_datasets.append(valid_ds)

if not train_datasets or not valid_datasets:
    raise ValueError("未能成功创建任何数据集，请检查HDF5文件路径和内容。")
    
full_train_dataset = ConcatDataset(train_datasets)
full_valid_dataset = ConcatDataset(valid_datasets)

train_loader = DataLoader(
    full_train_dataset, batch_size=params.BATCH_SIZE, shuffle=True, 
    num_workers=2, pin_memory=True, persistent_workers=True
)
valid_loader = DataLoader(
    full_valid_dataset, batch_size=params.BATCH_SIZE, shuffle=False, 
    num_workers=2, pin_memory=True, persistent_workers=True
)

print(f"数据集和DataLoader创建完成。")
print(f"  -> 训练样本数: {len(full_train_dataset)}")
print(f"  -> 验证样本数: {len(full_valid_dataset)}")
print("-" * 50)
```

这里由于FourcastNet的输入为一个720x1440的“图像”，而从CDS下载下来的数据集尺寸为721x1440，我们简单起见直接把最后最后一个维度删掉，事实上FourcastNet在训练时也是这么做的。

微调策略方面，我们选择了主流的Adam优化器，然后，我们选择的学习率为1e-5，由于Kaggle上的GPU时间有限，所以我们只是跑了5个epoch进行可行性的检测。
```python
best_valid_loss_full = float('inf')
finetuned_model_save_path_full = os.path.join(params.output_dir, 'finetuned_model_full_vars.pt')

logging.info("="*20 + " 开始全变量微调训练 " + "="*20)
for epoch in range(params.num_epochs):
    model.train()
    train_loss_epoch = 0
    for inp, tar in tqdm(train_loader, desc=f"Epoch {epoch+1}/{params.num_epochs} [Train]"):
        inp, tar = inp.to(params.device), tar.to(params.device)
        optimizer.zero_grad()
        output = model(inp)
        loss = criterion(output, tar)
        loss.backward()
        optimizer.step()
        train_loss_epoch += loss.item()
    avg_train_loss = train_loss_epoch / len(train_loader)

    model.eval()
    valid_loss_epoch = 0
    with torch.no_grad():
        for inp, tar in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{params.num_epochs} [Valid]"):
            inp, tar = inp.to(params.device), tar.to(params.device)
            output = model(inp)
            loss = criterion(output, tar)
            valid_loss_epoch += loss.item()
    avg_valid_loss = valid_loss_epoch / len(valid_loader)
    
    logging.info(f"Epoch {epoch+1}/{params.num_epochs} | 训练损失: {avg_train_loss:.6f} | 验证损失: {avg_valid_loss:.6f}")
    
    if avg_valid_loss < best_valid_loss_full:
        best_valid_loss_full = avg_valid_loss
        torch.save({'model_state': model.state_dict()}, finetuned_model_save_path_full)
        logging.info(f"发现更好的模型！验证损失降低至 {best_valid_loss_full:.6f}。模型已保存到 {finetuned_model_save_path_full}")

```
## iter 1 - 单变量 v.s. 多变量
后来我想了一下，能不能将预测的变量限制在单变量中，比如我想要训练一个专门预测地表温度的大模型，我们就通过给它提供只有t2m数据的数据集，在这个数据集上做训练
``` python
...
        output_t2m = output_all_vars[:, t2m_channel_index, :, :]
        target_t2m = tar[:, t2m_channel_index, :, :]
...
```
相较于iter0中的代码，我们只在这里做了修改。即，把20个通道过滤成一个通道。

我们将使用单变量进行微调的模型与使用全变量微调得到的模型进行比较，得到如下结果：

![](http://26l1b06988.qicp.vip:38000/pictures/20250610132045.png)

![](http://26l1b06988.qicp.vip:38000/pictures/20250610133836.png)

![](http://26l1b06988.qicp.vip:38000/pictures/20250610133852.png)

可以看到，虽然在一开始单变量预测的结果要略优于全变量预测，但是之后单变量的预测结果误差增长速度远快于全变量，推测可能是过拟合造成的。但事实上这两个模型的误差都远未达到可用的程度。
## iter2 - 层冻结策略

考虑气象预测这一特定问题的特点。气象预测的基础：物理规律长期来看是不会变动的，然而，人类活动可能会导致地球的平均温度发生变化，极端气候增加，这些是导致分布漂移的根本因素。深入分析FourcastNet的网络结构，我们发现其架构设计本身就隐含了对“不变物理规律”和“可变统计分布”的分层处理思想，这使得它非常适合通过特定的微调策略来适应气候变化带来的挑战。
所以，我们通过微调部分层来完成性能的改进，从上面的分析我们发现，由于单变量的微调可能会导致过拟合的出现，我选择采用全变量微调。从`network/afnonet.py`中我们得知，FourcastNet是一个12层结构的AFNO神经网络，和经典的CNN类似，由于越靠前的层接触到的数据越原始，因此越靠前的层越具有基础的知识（如基本的物理知识）越靠后的层越具有高层的，全局的知识（如全局气候情况）

![](http://26l1b06988.qicp.vip:38000/pictures/20250617143602.png)

在实践中，我采取冻结前面8个Block，微调后面4个Block的策略，使用2024年5月-7月的数据进行微调，取得了很不错的效果（验证选择2025年6月1日的数据开始预测）。

![](http://26l1b06988.qicp.vip:38000/pictures/20250617143928.png)

下图为60个epoch训练过程中的训练集和验证集Loss变化曲线

![](http://26l1b06988.qicp.vip:38000/pictures/20250619091415.png)

此外，我针对我使用不同的微调策略微调出来的模型做了一系列的比较，详细的比较内容已经放在了ppt中，这里我不使用微调后的模型与baseline（没有微调的模型）进行对比的原因是，原始模型由于分布偏移等原因误差过大，此处如果与原始模型进行比较的话没有任何实际的意义。
# MCP工具
此外，我们使用FastMCP为微调后的大模型提供了与LLM进行交互的接口，以提升我们工作的实用性。详见ppt内容。
# 异常点观察
在对生成的结果进行观察时，我还发现了有趣的现象：
## 预测的图像不平滑

![](http://26l1b06988.qicp.vip:38000/pictures/20250610131825.png)

注意这张图的左右两边，可以看出，模型预测出来的结果看起来“清晰度”没有原始数据那么高，换句话说，它看起来像是由一个个”色块“组成的，下面为原始图像和预测图像的对比：

![](http://26l1b06988.qicp.vip:38000/pictures/20250623000332.png)
![](http://26l1b06988.qicp.vip:38000/pictures/20250623000416.png)

而在原始的模型中，我们也没有能观察到这样的区别，查阅资料后，我发现问题可能出现在我们损失函数的选取上，由于我们的损失函数选择的是MSE，而MSE作为损失函数具有如下的特点：
- **倾向于产生“安全”的平均预测**：当模型面对不确定性时（例如，一个像素点可能是 A 值也可能是 B 值），为了最小化平方误差，模型的最优策略是预测 A 和 B 的平均值。这会导致模型避免做出极端但可能正确的预测（例如尖锐的边缘），而倾向于产生平滑、模糊的“折中”结果。
- **惩罚大错误**：MSE 对大的误差（离群点）给予非常大的惩罚。这使得模型会尽力避免在任何一个像素上犯大错，代价就是在很多像素上都犯一点小错，宏观上就表现为图像的模糊。
这两点共同作用，使得预测出来的图像看上去较为模糊。
## 在冷暖交界处误差异常增大

![](http://26l1b06988.qicp.vip:38000/pictures/20250610131825.png)

同样观察上面的那幅图，但这次我们关注冷暖交界的位置：

![](http://26l1b06988.qicp.vip:38000/pictures/20250623003820.png)

这幅图中，左边为真实情况，右边为预测结果，可以看到，右边的预测多出了许多气温异常高的点，再次查阅资料，我发现这是由于FourcastNet骨干网络选用AFNO作为算子造成的，AFNO的原理如下：它通过保留低频分量并截断或衰减高频分量来学习映射关系。当模型将处理后的频域信息通过逆傅里叶变换转回空间域（即生成图像）时，这种对高频信息的硬性截断，会导致在原始信号中存在**不连续点或剧烈变化**（如陆地与海洋的清晰边界、山脉的陡峭边缘）的邻近区域产生**高频振荡和过冲**，这样的现象又被称为**吉布斯现象**。
# 总结
## 经验教训
在完成这个大作业的过程中，我学习了许多微调模型相关的知识，同时也踩了不少初学者容易踩的坑。我从这次经历中得到的最深刻的教训之一就是，在进行正式的训练前，**一定要检查数据集是否完整**。
抛开这次”踩坑“经历不谈，我还认识到，成功的微调策略并非一成不变的“万金油”，而是需要深度结合对模型架构的理解和对问题领域的洞察。通过多次迭代，我发现：

1. **多变量联合预测优于单变量**: 简单的单变量微调虽然在初期表现尚可，但很快会因丢失变量间的物理关联性而陷入过拟合，导致长期预测能力迅速下降。这证明了FourCastNet利用多通道数据捕捉复杂气象系统耦合关系的必要性。
    
2. **分层微调是应对分布漂移的有效手段**: 本次项目中最成功的策略——冻结代表基础物理规律的底层网络、仅微调适应近期气候统计特征的高层网络——验证了一个核心思想：模型的不同部分承载着不同层次的知识。通过这种方式，我们既保留了模型泛化能力强的“物理先验”，又高效地让它适应了因气候变化带来的“统计漂移”，最终取得了远超全参数微调的优异成果。

此外，对模型异常输出的深入分析也让我收获颇丰。我们观察到的预测图像“清晰度”不足的问题，揭示了MSE损失函数在图像生成任务中倾向于产生“平均化”模糊结果的内在本性；而在冷暖交界处出现的“吉布斯现象”伪影，则让我对AFNO算子在处理不连续信号时的傅里叶变换局限性有了更直观的认识。

总而言之，这个项目不仅是一次成功的模型微调实践，更是一次从数据准备、理论分析到工程实现的全方位学习。它让我深刻理解到，前沿的人工智能技术在应用于天气预报等严肃科学领域时，必须建立在可靠的数据、对模型内在机理的深刻理解以及与领域知识紧密结合的智慧策略之上。
## 未来展望
尽管当前的微调模型已经可以完成基本的天气预测+LLM智能推荐功能，但是我们的应用仍然有很大的改进空间，初步设想有如下几个方面：
- 对损失函数进行改进，生成“分辨率”更高的图像。
- 优化MCP工具，为用户提供更详细的建议和意见，提高工具的可用性
- 优化微调策略，提高预测的准确性
- 使用持续构建脚本，保证模型的高可用性。
# 参考资料

[NVlabs/FourCastNet: Initial public release of code, data, and model weights for FourCastNet](https://github.com/NVlabs/FourCastNet)

[[2202.11214] FourCastNet: A Global Data-driven High-resolution Weather Model using Adaptive Fourier Neural Operators](https://arxiv.org/abs/2202.11214)

[climatechange-ai-tutorials/fourcastnet: Learn how to use FourCastNet, a weather model based on deep learning, to obtain short to medium-range forecasts of crucial atmospheric variables such as surface wind velocities.](https://github.com/climatechange-ai-tutorials/fourcastnet)

[FourCastNet - PaddleScience Docs](https://paddlescience-docs.readthedocs.io/zh-cn/latest/zh/examples/fourcastnet/#31)

[Climate Data Store](https://cds.climate.copernicus.eu/)

[欢迎使用 FastMCP 2.0！ - FastMCP --- Welcome to FastMCP 2.0! - FastMCP](https://gofastmcp.com/getting-started/welcome)
