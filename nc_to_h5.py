# BSD 3-Clause License
#
# Copyright (c) 2022, FourCastNet authors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# This sequential script is a modification based on the original FourCastNet code.

import h5py
import numpy as np
import time
from netCDF4 import Dataset as DS
import os

# --- 配置: 定义从源变量到HDF5通道的映射 ---
# 格式: (文件类型, nc变量名, h5通道索引, 气压层索引)
# 文件类型: 'sfc' 代表地表层文件, 'pl' 代表等压面文件
# 气压层索引: 'pl' 文件中 'level' 维度的索引。对于 'sfc' 文件，此项为 None
# 根据原始脚本:
# level=0 -> 50 hPa, 1 -> 500 hPa, 2 -> 850 hPa, 3 -> 1000 hPa
VARIABLE_MAP = [
    # 来自 _sfc.nc 的地表变量
    ('sfc', 'u10',   0, None),  # 10米U-风
    ('sfc', 'v10',   1, None),  # 10米V-风
    ('sfc', 't2m',   2, None),  # 2米温度
    ('sfc', 'sp',    3, None),  # 地表气压
    ('sfc', 'msl',   4, None),  # 海平面气压
    ('sfc', 'tcwv', 19, None),  # 总可降水量

    # 来自 _pl.nc 的等压面变量
    ('pl',  't',     5, 2),     # 850 hPa 温度
    ('pl',  'u',     6, 3),     # 1000 hPa U-风
    ('pl',  'v',     7, 3),     # 1000 hPa V-风
    ('pl',  'z',     8, 3),     # 1000 hPa 位势高度
    ('pl',  'u',     9, 2),     # 850 hPa U-风
    ('pl',  'v',    10, 2),     # 850 hPa V-风
    ('pl',  'z',    11, 2),     # 850 hPa 位势高度
    ('pl',  'u',    12, 1),     # 500 hPa U-风
    ('pl',  'v',    13, 1),     # 500 hPa V-风
    ('pl',  'z',    14, 1),     # 500 hPa 位势高度
    ('pl',  't',    15, 1),     # 500 hPa 温度
    ('pl',  'z',    16, 0),     # 50 hPa 位势高度
    ('pl',  'r',    17, 1),     # 500 hPa 相对湿度
    ('pl',  'r',    18, 2),     # 850 hPa 相对湿度
]
# 在HDF5文件中要创建的总通道数
NUM_CHANNELS = 20

def writetofile_sequential(src_path, f_dest_handle, channel_idx, var_name, Nimgtot,
                           src_level_idx=None, batch_size=64):
    """
    从源NetCDF文件读取一个变量，并将其写入已打开的目标HDF5文件的特定通道。
    """
    print(f"  正在处理通道 {channel_idx} ('{var_name}')...", end='', flush=True)
    start_time = time.time()

    # 检查源文件
    if not os.path.isfile(src_path):
        print(f"\n错误: 源文件未找到: {src_path}")
        return False

    # 打开源文件
    with DS(src_path, 'r', format="NETCDF4") as f_src_ds:
        f_src_var = f_src_ds.variables[var_name]
        
        # 逐批次读取和写入数据，以节约内存
        idx = 0
        while idx < Nimgtot:
            read_count = min(batch_size, Nimgtot - idx)
            
            # 从源文件读取一批数据
            if len(f_src_var.shape) == 4:  # 等压面数据
                ims = f_src_var[idx : idx + read_count, src_level_idx]
            else:  # 地表数据
                ims = f_src_var[idx : idx + read_count]
            
            # 将数据批次写入目标HDF5文件
            f_dest_handle['fields'][idx : idx + read_count, channel_idx, :, :] = ims
            idx += read_count

    elapsed_time = time.time() - start_time
    print(f" 完成，耗时 {elapsed_time:.2f} 秒。")
    return True

def main():
    """
    主函数，驱动交互式数据转换过程。
    """
    print("--- FourCastNet HDF5 数据转换工具 (单进程版) ---")
    
    # --- 1. 获取用户输入 ---
    src_dir = input("请输入包含 .nc 文件的源目录: ")
    dest_dir = input("请输入用于存放 .h5 文件的目标目录: ")
    base_filename = input("请输入基础文件名 (例如 'oct_2021_19_31'): ")

    # 验证路径
    if not os.path.isdir(src_dir):
        print(f"错误: 源目录未找到: {src_dir}")
        return
    if not os.path.isdir(dest_dir):
        try:
            os.makedirs(dest_dir, exist_ok=True)
            print(f"提示: 目标目录 {dest_dir} 不存在，已自动创建。")
        except OSError as e:
            print(f"错误: 无法创建目标目录: {dest_dir} - {e}")
            return

    # --- 2. 构建文件路径并确定数据维度 ---
    src_sfc_path = os.path.join(src_dir, f"{base_filename}_sfc.nc")
    src_pl_path = os.path.join(src_dir, f"{base_filename}_pl.nc")
    dest_path = os.path.join(dest_dir, f"{base_filename}.h5")

    print("\n正在检查源文件并确定数据维度...")
    try:
        with DS(src_sfc_path, 'r') as f:
            var_shape = f.variables['u10'].shape
            Nimgtot = var_shape[0]
            img_shape = var_shape[1:]
            print(f"  - 发现 {Nimgtot} 个时间步 (图像)。")
            print(f"  - 图像分辨率: {img_shape[0]}x{img_shape[1]}。")
    except FileNotFoundError:
        print(f"致命错误: 在 {src_sfc_path} 未找到地表文件")
        return
    except KeyError:
        print(f"致命错误: 在 {src_sfc_path} 中未找到变量 'u10'。无法确定数据维度。")
        return

    # --- 3. 检查目标文件并准备写入 ---
    if os.path.exists(dest_path):
        overwrite = input(f"文件 {dest_path} 已存在。是否覆盖? (y/n): ").lower()
        if overwrite != 'y':
            print("操作中止。")
            return
            
    print("\n开始数据转换...")
    total_start_time = time.time()

    # --- 4. 创建HDF5文件并逐个写入所有变量 ---
    try:
        with h5py.File(dest_path, 'w') as f_dest:
            print(f"已创建目标文件: {dest_path}")
            
            # 创建主数据集以容纳所有变量
            f_dest.create_dataset('fields', 
                                  shape=(Nimgtot, NUM_CHANNELS, img_shape[0], img_shape[1]), 
                                  dtype=np.float32)
            print("数据集 'fields' 已创建。")
            
            # 按通道索引排序，确保写入顺序正确
            sorted_map = sorted(VARIABLE_MAP, key=lambda x: x[2])
            
            all_successful = True
            for file_type, var_name, channel_idx, pl_idx in sorted_map:
                if file_type == 'sfc':
                    src_path = src_sfc_path
                    src_level_idx = None
                else: # 'pl'
                    src_path = src_pl_path
                    src_level_idx = pl_idx
                
                if not writetofile_sequential(src_path, f_dest, channel_idx, var_name, Nimgtot, src_level_idx):
                    all_successful = False
                    break # 如果有任何一个文件处理失败，则中止

            if all_successful:
                total_end_time = time.time()
                print(f"\n--- 转换完成 ---")
                print(f"总耗时: {total_end_time - total_start_time:.2f} 秒。")
                print(f"数据已保存至: {dest_path}")
            else:
                print("\n--- 转换因错误而中止 ---")

    except Exception as e:
        print(f"\n处理过程中发生严重错误: {e}")

if __name__ == "__main__":
    main()
    