# 环境配置方法

## 各个文件的作用

- finetune.py 层冻结微调策略的微调脚本
- model-test.ipynb 测试模型预测误差的可视化脚本，同时测试路径配置是否正确，这里的路径配置和mcp-server/main.py中的配置是一样的
- nc_downloader.py 从ECWMF下载.nc数据文件的交互式脚本
- nc_to_h5.py 将下载下来的nc文件转换成模型可以读取的h5文件时需要使用的脚本
- mcp-server/main.py MCP服务器的主要逻辑

## 测试脚本
路径配置写在notebook里面了

## MCP服务器
### 前置要求
1. 安装python3.11或以上（建议使用uv包管理）
2. Cherry Studio v1.4.3
3. 从Nvidia的官方仓库中clone FourcastNet官方仓库
4. 从Globus中下载官方的预训练文件

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
