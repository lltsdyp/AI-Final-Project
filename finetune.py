import os
import h5py
import time
import numpy as np
import torch
import torch.nn as nn
import sys
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from collections import OrderedDict

# Get the model config from default configs
sys.path.insert(1, './FourCastNet/') # insert code repo into path
from utils.YParams import YParams

# 确保networks/afnonet.py可用
try:
    from networks.afnonet import AFNONet
except ImportError:
    print("="*50)
    print("错误：无法找到 'networks/afnonet.py'。")
    print("请确保你已经将FourcastNet的 'networks' 目录上传到Kaggle环境，")
    print("或者将其添加为Utility Script。")
    print("="*50)
    raise

torch.autograd.set_detect_anomaly(True)
class Config:
    # --- 路径配置 ---
    H5_PATHS = {
        '2024-01': '/kaggle/input/era5-h5-2024-01/output_h5/2024-01.h5',
        # ...添加更多...
    }
    PRETRAINED_MODEL_PATH = '/kaggle/input/fourcastnet/pytorch/default/1/ccai_demo/model_weights/FCN_weights_v0/backbone.ckpt'
    STATS_DIR = '/kaggle/input/fourcastnet/pytorch/default/1/ccai_demo/additional/stats_v0'
    
    OUTPUT_DIR = '/kaggle/working/finetuned_fcn_robust_freezing'
    FINETUNED_MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, 'finetuned_model_best.pt')

    # --- 数据配置 ---
    CHANNELS = list(range(20))
    TRAIN_DAYS_PER_FILE = 25
    VALID_START_DAY_PER_FILE = 25
    TIMESTEPS_PER_DAY = 4

    # --- 模型架构配置 (必须与预训练模型 backbone.ckpt 完全匹配) ---
    IMG_SIZE = (720, 1440)
    PATCH_SIZE = 8
    N_IN_CHANNELS = 20
    N_OUT_CHANNELS = 20
    NUM_BLOCKS = 8 # 预训练模型使用了8个块
    MODEL_DEPTH = 12 
    FREEZE_BLOCKS_N = 8
    EMBED_DIM = 768
    USE_OROGRAPHY = False

    # --- 微调超参数 ---
    NUM_EPOCHS = 60
    BATCH_SIZE = 1
    LEARNING_RATE = 1e-5
    EARLY_STOPPING_PATIENCE = 5
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 实例化配置
params = Config()
os.makedirs(params.OUTPUT_DIR, exist_ok=True)
print(f"配置加载完成。设备: {params.DEVICE}")
print(f"输出将保存在: {params.OUTPUT_DIR}")
print("-" * 50)

def load_pretrained_model(model, checkpoint_path, device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device,weights_only=False)
        state_dict = checkpoint.get('model_state', checkpoint)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        print(f"成功从 {checkpoint_path} 加载预训练模型权重。")
    except Exception as e:
        print(f"加载预训练模型失败: {e}")
        raise
    return model

class ModelConfig:
    def __init__(self, params):
        self.patch_size = params.PATCH_SIZE
        self.N_in_channels = params.N_IN_CHANNELS
        self.N_out_channels = params.N_OUT_CHANNELS
        self.num_blocks = params.NUM_BLOCKS

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
            print(f" [警告] 输入数据包含 NaN 或 Inf (idx={idx}, input_idx={input_idx})")
            # 可选：替换为零或跳过该样本
            inp_raw = torch.zeros_like(inp_raw)
            abort()
    
        if torch.isnan(tar_raw).any() or torch.isinf(tar_raw).any():
            print(f" [警告] 目标数据包含 NaN 或 Inf (idx={idx}, target_idx={target_idx})")
            tar_raw = torch.zeros_like(tar_raw)
            abort()
    
        # 归一化
        inp_norm = (inp_raw - self.means) / (self.stds + 1e-6)
        tar_norm = (tar_raw - self.means) / (self.stds + 1e-6)
    
        # 再次检查归一化后的数据
        if torch.isnan(inp_norm).any() or torch.isinf(inp_norm).any():
            print(f" [警告] 归一化后输入数据包含 NaN 或 Inf (idx={idx})")
            inp_norm = torch.zeros_like(inp_norm)
            abort()
    
        if torch.isnan(tar_norm).any() or torch.isinf(tar_norm).any():
            print(f" [警告] 归一化后目标数据包含 NaN 或 Inf (idx={idx})")
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




print("正在实例化AFNONet模型...")
model_config = ModelConfig(params)
# 注意：AFNONet的depth参数默认为12，但我们的预训练模型实际上只有8个块。
# 官方的训练脚本中，depth参数似乎与num_blocks是独立的，但这里的blocks列表长度是由depth决定的。
# 根据 `self.blocks` 的定义，其长度为 `depth`。而我们的 `NUM_BLOCKS` 是 8，这可能意味着 `depth` 应该是8。
# 我们将假设 `depth` 应该等于 `NUM_BLOCKS`。
model = AFNONet(
    params=model_config,
    img_size=params.IMG_SIZE,
    embed_dim=params.EMBED_DIM,
    depth=params.MODEL_DEPTH # 确保块的数量与配置一致
).to(params.DEVICE)

print("模型实例化完成。")
model = load_pretrained_model(model, params.PRETRAINED_MODEL_PATH, params.DEVICE)

def apply_layer_freezing(model, num_blocks_to_freeze):
    """
    冻结模型的部分层，只保留最后几层用于微调。
    """
    print("\n应用层冻结策略...")
    
    # 1. 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
        
    # 2. 解冻需要微调的层
    unfrozen_layers = []
    
    # 解冻后期的 Transformer 块 (self.blocks)
    if num_blocks_to_freeze < model.params.num_blocks:
        for i in range(num_blocks_to_freeze, model.params.num_blocks):
            for param in model.blocks[i].parameters():
                param.requires_grad = True
        unfrozen_layers.append(f'blocks[{num_blocks_to_freeze}:]')
        
    # 解冻最后的归一化层 (self.norm)
    for param in model.norm.parameters():
        param.requires_grad = True
    unfrozen_layers.append('norm')

    # 解冻输出头 (self.head)
    for param in model.head.parameters():
        param.requires_grad = True
    unfrozen_layers.append('head')
    
    print(f"以下层将被微调: {', '.join(unfrozen_layers)}")
    
    # 打印参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params / 1e6:.2f}M")
    print(f"可训练参数量: {trainable_params / 1e6:.2f}M ({trainable_params * 100 / total_params:.2f}%)")
    
    return model

model = apply_layer_freezing(model, params.FREEZE_BLOCKS_N)

trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(trainable_params, lr=params.LEARNING_RATE)
criterion = nn.MSELoss()

steps_per_epoch = len(train_loader)
total_steps = params.NUM_EPOCHS * steps_per_epoch

print("\n模型、优化器和损失函数已准备就绪 (优化器只针对可训练层)。")
print("-" * 50)

best_val_loss = float('inf')
start_time = time.time()

# --- 新增：早停相关的变量 ---
epochs_no_improve = 0  # 记录验证损失没有改善的连续epoch数

print("开始微调循环 ...")
# 主循环，会因为早停而提前中断
for epoch in range(params.NUM_EPOCHS):
    epoch_start_time = time.time()
    
    # --- 训练阶段 ---
    model.train()
    train_loss = 0.0
    
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(params.DEVICE), targets.to(params.DEVICE)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item()
        
        if (i + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{params.NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], Current Loss (MSE): {loss.item():.6f}')
            
    avg_train_loss = train_loss / len(train_loader)

    # --- 验证阶段 ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in valid_loader:
            inputs, targets = inputs.to(params.DEVICE), targets.to(params.DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(valid_loader)
    
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    
    # --- 打印结果 ---
    print("-" * 50)
    print(f'Epoch [{epoch+1}/{params.NUM_EPOCHS}] 完成 | 用时: {epoch_duration:.2f}s')
    print(f'  -> 平均训练损失 (MSE): {avg_train_loss:.6f}')
    print(f'  -> 平均验证损失 (MSE): {avg_val_loss:.6f}')

    # --- 保存最佳模型与早停逻辑 ---
    if avg_val_loss < best_val_loss:
        print(f'  -> 验证损失从 {best_val_loss:.6f} 改善到 {avg_val_loss:.6f}，模型已保存。')
        best_val_loss = avg_val_loss
        epochs_no_improve = 0  # 重置计数器
        
        torch.save({
            'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_val_loss,
        }, params.FINETUNED_MODEL_SAVE_PATH)
    else:
        epochs_no_improve += 1  # 增加计数器
        print(f'  -> 验证损失没有改善，当前最佳为 {best_val_loss:.6f}。')
        print(f'  -> 早停计数: {epochs_no_improve}/{params.EARLY_STOPPING_PATIENCE}')

    print("-" * 50)

    # --- 检查是否触发早停 ---
    if epochs_no_improve >= params.EARLY_STOPPING_PATIENCE:
        print(f"验证损失已连续 {params.EARLY_STOPPING_PATIENCE} 个epoch没有改善，触发早停。")
        break  # 中断训练循环

end_time = time.time()
total_duration = end_time - start_time

print("微调完成！")
print(f"总用时: {total_duration:.2f}秒")
print(f"最佳验证损失 (MSE): {best_val_loss:.6f} (在第 {epoch + 1 - epochs_no_improve} 个epoch达到)")
print(f"最终模型保存在: {params.FINETUNED_MODEL_SAVE_PATH}")
