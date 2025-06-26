# 环境配置方法

## 各个文件的作用

- finetune.py 层冻结微调策略的微调脚本
- model-test.ipynb 测试模型预测误差的可视化脚本，同时测试路径配置是否正确，这里的路径配置和mcp-server/main.py中的配置是一样的
- nc_downloader.py 从ECWMF下载.nc数据文件的交互式脚本
- nc_to_h5.py 将下载下来的nc文件转换成模型可以读取的h5文件时需要使用的脚本
- mcp-server/main.py MCP服务器的主要逻辑

## 数据集下载
调用nc_downloader.py，该脚本会用交互式的方式来下载相应的数据集
接着调用nc_to_h5.py，该脚本同样使用交互式的方式将Surface数据和Pressure Level数据进行合并，脚本中的基础文件名指的是下载下来的文件去掉_sfc.nc或者_pl.nc之后剩余的前缀

我在百度云上传了一个演示用数据集
通过网盘分享的文件：era5_2025_06.h5
链接: https://pan.baidu.com/s/15GXxKphvXwgTbA8fQvD2sw?pwd=3e3s 提取码: 3e3s

## 微调模型权重下载
上传到Huggingface中了
https://huggingface.co/lltsdyp/Finetuned-FCN

## 测试脚本
路径配置写在notebook里面了

## MCP服务器
### 前置要求
1. 安装python3.11或以上（建议使用uv包管理）
2. Cherry Studio v1.4.3或者其他可用的大模型调用客户端
3. 从Nvidia的官方仓库中clone FourcastNet官方仓库
4. 从Globus中下载官方的预训练文件

CherryStudio下载地址：[Cherry Studio官网](https://cherrystudiocn.com/)

3 4步参考[FourcastNet官方仓库](https://github.com/NVlabs/FourCastNet)

### 配置预训练权重，归一化数据
**严格**按照notebook中进行，
需要修改notebook中的路径以及mcp-server/main.py中的路径

### 设置步骤
首先在Cherry Studio配置可接入MCP的大模型（如Gemini）
然后按照下图所示的方法配置Gemini接入我们的WeatherForecast MCP服务器
![](http://26l1b06988.qicp.vip:38000/pictures/20250617203449.png)

下面的参数填写：
``` plaintext
--directory
<Your git base directory>/mcp-server
run
main.py
```

确认对话窗口开启了MCP服务器

![](http://26l1b06988.qicp.vip:38000/pictures/20250617203702.png)

## 引用
FourcastNet:
```
@article{pathak2022fourcastnet,
  title={Fourcastnet: A global data-driven high-resolution weather model using adaptive fourier neural operators},
  author={Pathak, Jaideep and Subramanian, Shashank and Harrington, Peter and Raja, Sanjeev and Chattopadhyay, Ashesh and Mardani, Morteza and Kurth, Thorsten and Hall, David and Li, Zongyi and Azizzadenesheli, Kamyar and Hassanzadeh, Pedram and Kashinath, Karthik and Anandkumar, Animashree},
  journal={arXiv preprint arXiv:2202.11214},
  year={2022}
}
```
