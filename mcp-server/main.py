# server.py
import datetime
from typing import Any, Dict, List, Optional, OrderedDict
from mcp.server.fastmcp import FastMCP,Image
from PIL import Image as PILImage


import xarray as xr
import numpy as np
import os, sys, time
import numpy as np
import h5py
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt


from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
sys.path.insert(1, 'E:/FourCastNet/') # insert code repo into path
from networks.afnonet import AFNONet
from utils.YParams import YParams

# data and model paths
current_script_path=os.path.dirname(os.path.abspath(__file__))
# data_path = "E:/ccai_demo/data/FCN_ERA5_data_v0/out_of_sample"
data_path = "D:/NCFiles/output"
model_path = "E:/ccai_demo/model_weights/FCN_weights_v0/backbone.ckpt"
global_means_path = "E:/ccai_demo/additional/stats_v0/global_means.npy"
global_stds_path = "E:/ccai_demo/additional/stats_v0/global_stds.npy"
time_means_path = "E:/ccai_demo/additional/stats_v0/time_means.npy"
land_sea_mask_path = "E:/ccai_demo/additional/stats_v0/land_sea_mask.npy"
config_file = "E:/FourCastNet/config/AFNO.yaml"
config_name = "afno_backbone"
model_key="model_state"
params = YParams(config_file, config_name)
finetuned_checkpoint_path=r'E:\pending_models\finetuned_model_best.pt'
user_agent = [
    "Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_6_8; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 "
    "Safari/534.50",
    "Mozilla/5.0 (Windows; U; Windows NT 6.1; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 "
    "Safari/534.50",
    "Mozilla/5.0 (Windows NT 10.0; WOW64; rv:38.0) Gecko/20100101 Firefox/38.0",
    "Mozilla/5.0 (Windows NT 10.0; WOW64; Trident/7.0; .NET4.0C; .NET4.0E; .NET CLR 2.0.50727; .NET CLR "
    "3.0.30729; .NET CLR 3.5.30729; InfoPath.3; rv:11.0) like Gecko",
    "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0)",
    "Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0)",
    "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1)",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.6; rv:2.0.1) Gecko/20100101 Firefox/4.0.1",
    "Mozilla/5.0 (Windows NT 6.1; rv:2.0.1) Gecko/20100101 Firefox/4.0.1",
    "Opera/9.80 (Macintosh; Intel Mac OS X 10.6.8; U; en) Presto/2.8.131 Version/11.11",
    "Opera/9.80 (Windows NT 6.1; U; en) Presto/2.8.131 Version/11.11",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_0) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 "
    "Safari/535.11",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Maxthon 2.0)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; TencentTraveler 4.0)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; The World)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Trident/4.0; SE 2.X MetaSr 1.0; SE 2.X MetaSr 1.0; .NET "
    "CLR 2.0.50727; SE 2.X MetaSr 1.0)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; 360SE)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Avant Browser)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)",
    "Mozilla/5.0 (iPhone; U; CPU iPhone OS 4_3_3 like Mac OS X; en-us) AppleWebKit/533.17.9 (KHTML, like Gecko) "
    "Version/5.0.2 Mobile/8J2 Safari/6533.18.5",
    "Mozilla/5.0 (iPod; U; CPU iPhone OS 4_3_3 like Mac OS X; en-us) AppleWebKit/533.17.9 (KHTML, like Gecko) "
    "Version/5.0.2 Mobile/8J2 Safari/6533.18.5",
    "Mozilla/5.0 (iPad; U; CPU OS 4_3_3 like Mac OS X; en-us) AppleWebKit/533.17.9 (KHTML, like Gecko) "
    "Version/5.0.2 Mobile/8J2 Safari/6533.18.5",
    "Mozilla/5.0 (Linux; U; Android 2.3.7; en-us; Nexus One Build/FRF91) AppleWebKit/533.1 (KHTML, like Gecko) "
    "Version/4.0 Mobile Safari/533.1",
    "MQQBrowser/26 Mozilla/5.0 (Linux; U; Android 2.3.7; zh-cn; MB200 Build/GRJ22; CyanogenMod-7) "
    "AppleWebKit/533.1 (KHTML, like Gecko) Version/4.0 Mobile Safari/533.1",
    "Opera/9.80 (Android 2.3.4; Linux; Opera Mobi/build-1107180945; U; en-GB) Presto/2.8.149 Version/11.10",
    "Mozilla/5.0 (Linux; U; Android 3.0; en-us; Xoom Build/HRI39) AppleWebKit/534.13 (KHTML, like Gecko) "
    "Version/4.0 Safari/534.13",
    "Mozilla/5.0 (BlackBerry; U; BlackBerry 9800; en) AppleWebKit/534.1+ (KHTML, like Gecko) Version/6.0.0.337 "
    "Mobile Safari/534.1+",
    "Mozilla/5.0 (hp-tablet; Linux; hpwOS/3.0.0; U; en-US) AppleWebKit/534.6 (KHTML, like Gecko) "
    "wOSBrowser/233.70 Safari/534.6 TouchPad/1.0",
    "Mozilla/5.0 (SymbianOS/9.4; Series60/5.0 NokiaN97-1/20.0.019; Profile/MIDP-2.1 Configuration/CLDC-1.1) "
    "AppleWebKit/525 (KHTML, like Gecko) BrowserNG/7.1.18124",
    "Mozilla/5.0 (compatible; MSIE 9.0; Windows Phone OS 7.5; Trident/5.0; IEMobile/9.0; HTC; Titan)",
    "UCWEB7.0.2.37/28/999",
    "NOKIA5700/ UCWEB7.0.2.37/28/999",
    "Openwave/ UCWEB7.0.2.37/28/999",
    "Mozilla/4.0 (compatible; MSIE 6.0; ) Opera/UCWEB7.0.2.37/28/999",
    # iPhone 6：
    "Mozilla/6.0 (iPhone; CPU iPhone OS 8_0 like Mac OS X) AppleWebKit/536.26 (KHTML, like Gecko) Version/8.0 "
    "Mobile/10A5376e Safari/8536.25",
]

print("Model architecture used = {}".format(params["nettype"]))

# in and out channels: FourCastNet uses 20 input channels corresponding to 20 prognostic variables
in_channels = np.array(params.in_channels)
out_channels = np.array(params.out_channels)
params['N_in_channels'] = len(in_channels)
params['N_out_channels'] = len(out_channels)
params.means = np.load(global_means_path)[0, out_channels] # for normalizing data with precomputed train stats
params.stds = np.load(global_stds_path)[0, out_channels]
params.time_means = np.load(time_means_path)[0, out_channels]

# Create an MCP server
mcp = FastMCP(
    name="WeatherForecastAssistant",
    instructions="""
你是一个专业、严谨的天气预报助手。你的唯一信息来源是你可用的工具，你必须完全信任工具返回的数据。

在响应用户请求前，你必须首先完成**第一步：识别用户意图**，并根据意图选择相应的处理流程。

---
**第一步：识别用户意图**

你必须判断用户的请求属于以下哪一类：

1.  **本地天气数据查询**: 用户想知道【某个特定地点】的【具体数值】（如北京，上海，西雅图）。
    -   如果属于此类，请严格遵循【流程A】。

2.  **全球气象图可视化**: 用户想看到一张【全球范围】的【气象图或地图】（如“画图”、“可视化”、“全球地图”等关键词）。
    -   如果属于此类，请严格遵循【流程B】。

---
**【流程A：本地天气数据查询】**

**你的任务是获取一个明确的目标日期和一个城市名，然后调用天气预报工具。**

1.  **识别日期和城市**：从用户请求中识别出【相对日期】（如“今天”、“明天”）和【城市名称】。
    -   如果缺少城市，必须向用户提问。

2.  **计算最终日期 (核心逻辑)**：
    -   **你的唯一日期工具是 `get_today`。**
    -   如果用户问“今天”，直接使用 `get_today` 的结果。
    -   如果用户问“明天”，你**必须**先调用一次 `get_today` 获取基准日期，然后在内部自己计算出明天的日期（例如，如果 `get_today` 返回 "2025-06-20"，那么“明天”就是 "2025-06-21"）。
    -   如果用户问“后天”，逻辑同上，在 `get_today` 的结果上加两天。
    -   **【绝对禁止】**：禁止在没有 `get_today` 作为基础的情况下凭空猜测日期。禁止向用户反问“今天是哪一天？”。

3.  **调用核心天气工具**：
    -   当你同时获得了【最终计算出的日期】和【城市名】后，**立即调用 `get_weather_forecast` 工具**。
    -   对于同一个用户请求，此工具只允许调用一次。

4.  **回答用户**：
    -   整合 `get_weather_forecast` 工具返回的完整数据，清晰地回答用户。

**示例思考过程 (对于“明天北京天气”):**
1.  用户意图是本地查询。关键词：“明天”、“北京”。
2.  我需要计算“明天”的日期。我的手册说必须用 `get_today`。
3.  调用 `get_today()` -> 返回 "2025-06-20"。
4.  我在脑中计算：明天 = "2025-06-20" + 1天 = "2025-06-21"。
5.  我现在有日期 "2025-06-21" 和城市 "北京"。
6.  材料齐全，立即调用 `get_weather_forecast(year=2025, month=6, day=21, city_name='北京')`。
7.  收到预报结果，整理后回答用户。


---
**【流程B：全球气象图可视化】**

1.  **确定日期 (自主行动)**:
    -   **你的首要行动是确定目标日期。** 如果用户的请求包含“今天”、“明天”等相对时间，你**必须立即、主动地、无需向用户确认地调用 `get_today` 工具**。
    -   使用 `get_today` 返回的日期作为基准，在内部计算出用户请求的最终日期（例如，“明天”）。
    -   **【绝对禁止】**: 绝对禁止向用户反问“今天是哪一天”。获取当前日期是你自己的责任，必须通过调用工具来完成。
2.  **确定变量**:
    -   从用户的请求中解析出他们想要可视化的气象变量名，并匹配到 `draw` 工具 `varname` 参数最接近的选项。
3.  **调用 `draw` 工具**:
    -   使用你计算出的最终日期和确定的变量名，调用 `draw` 工具。
    -   **【绝对禁止】**: 在此流程中，绝对禁止向用户询问城市名称。
4.  **展示图片**:
    -   `draw` 工具会返回一个图片路径，你的任务是向用户展示这张图片。

---
你的所有行为都必须严格遵循上述分类和流程。
    """,
)
device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
# move normalization tensors to gpu
# load time means: represents climatology
img_shape_x = 720
img_shape_y = 1440
variables = ['u10',
             'v10',
             't2m',
             'sp',
             'msl',
             't850',
             'u1000',
             'v1000',
             'z1000',
             'u850',
             'v850',
             'z850',
             'u500',
             'v500',
             'z500',
             't500',
             'z50' ,
             'r500',
             'r850',
             'tcwv']


# means and stds over training data
means = params.means
stds = params.stds

def get_data_for_date(year, month, day):
    # 在真实的系统中，这里会根据年月日找到正确的文件
    # 这里做了简化
    data_file_path = os.path.join(data_path, "era5_2025_06.h5")
    
    # 同样出于演示目的进行简化
    initial_condition_timestep = (day - 20) * 4 + 1
    
    return data_file_path, initial_condition_timestep

def load_model(model, params, checkpoint_file,use_finetuned=True):
    ''' helper function to load model weights '''
    checkpoint_fname = checkpoint_file
    checkpoint = torch.load(checkpoint_fname,weights_only=False)
    try:
        ''' FourCastNet is trained with distributed data parallel
            (DDP) which prepends 'module' to all keys. Non-DDP
            models need to strip this prefix '''
        new_state_dict = OrderedDict()
        for key, val in checkpoint[model_key].items():
            name = key[7:]
            if name != 'ged':
                new_state_dict[name] = val
        model.load_state_dict(new_state_dict)
    except:
        model.load_state_dict(checkpoint[model_key])
        
    if not use_finetuned:
        return model

    try:
        finetuned_checkpoint = torch.load(finetuned_checkpoint_path, weights_only=False)
        
        # 从保存结构中提取模型的 state_dict
        finetuned_state_dict = finetuned_checkpoint.get('model_state')

        if finetuned_state_dict is None:
            raise KeyError("在微调检查点文件中找不到 'model_state' 键。")

        # 将微调后的权重加载到模型上。
        # 这一步会用新权重覆盖掉那些在微调中被训练的层（如 norm, head）。
        # 因为 finetuned_state_dict 包含了所有层的参数，所以可以用 strict=True
        model.load_state_dict(finetuned_state_dict, strict=False)
        print("     微调权重覆盖成功。")

    except Exception as e:
        print(f"     错误：加载微调权重时出错: {e}")
        raise
    model.eval() # set to inference mode
    return model

if params.nettype == 'afno':
    model = AFNONet(params).to(device)  # 命名为 base_model 以示区分
else:
    raise Exception("not implemented")

# 2. 加载原始的预训练权重到基础模型 
model = load_model(model, params, model_path)
model = model.to(device)

def preprocess(data_file: str):
    # 只加载初始条件 (initial condition)
    data = h5py.File(data_file, 'r')['fields'][0, in_channels, 0:img_shape_x]
    data = np.expand_dims(data, axis=0) # 增加一个 batch 维度, shape: (1, 20, 720, 1440)
    
    # 标准化数据
    data = (data - means) / stds
    
    # 转换为 tensor
    data = torch.as_tensor(data).to(device, dtype=torch.float)
    return data

def inference_global(initial_condition, model, prediction_length):

    predicted = None
    with torch.no_grad():
        for i in range(prediction_length):
            # 1. 使用当前输入进行预测
            # model的输出shape是 (1, C, H, W)
            next_prediction = model(initial_condition)
            
            
            # b. 添加到结果列表
            predicted = next_prediction
            
            # 3. 准备下一次循环的输入
            # 将当前的预测结果作为下一次迭代的输入
            initial_condition = next_prediction
        
    prediction_cpu = predicted.cpu().numpy()

    return prediction_cpu

def inference(initial_condition, model, prediction_length, t2m_channel_index, lat_idx, lon_idx):
    """
    Performs autoregressive inference.
    
    Args:
        initial_condition (torch.Tensor): The starting data tensor, shape (1, C, H, W).
        model: The trained model.
        prediction_length (int): How many steps to predict into the future (e.g., 4 for 24 hours).
        t2m_channel_index (int): The index of the 't2m' variable.
        lat_idx (int): Latitude index for the target city.
        lon_idx (int): Longitude index for the target city.
        
    Returns:
        List[float]: A list of predicted temperatures for the given location.
    """
    
    predicted_temps = []
    current_input = initial_condition

    with torch.no_grad():
        for i in range(prediction_length):
            # 1. 使用当前输入进行预测
            # model的输出shape是 (1, C, H, W)
            next_prediction = model(current_input)
            
            # 2. 提取我们关心的 t2m 温度值
            # a. 从预测结果中获取 t2m 通道的数据
            t2m_prediction_normalized = next_prediction[0, t2m_channel_index, lat_idx, lon_idx]
            
            # b. 添加到结果列表
            predicted_temps.append(t2m_prediction_normalized.item())
            
            # 3. 准备下一次循环的输入
            # 将当前的预测结果作为下一次迭代的输入
            current_input = next_prediction

    return predicted_temps

@mcp.tool()
def inference_relative_humidity(year: int, month: int, day: int, city_name: str) -> List[float]:
    """
    获取指定城市在指定日期的近似相对湿度预报。
    该数据基于850hPa（约1.5公里高空）的相对湿度，可作为地面湿度的参考。
    返回值为6小时为间隔的相对湿度百分比。

    Args:
        year (int): 年份
        month (int): 月份
        day (int): 日
        city_name (str): 城市名称

    Returns:
        List[float]: 一个包含4个浮点数的列表，代表当天的相对湿度百分比(%)。
                     如果无法获取坐标，则返回空列表。
    """
    try:
        lat, lon = get_city_coordinates(city_name)
        if lat is None or lon is None:
            return []
    except Exception as e:
        print(f"获取坐标时出错: {e}")
        return []

    lat_idx = int((90 - lat) / 0.25)
    lon_idx = int(lon / 0.25) % 1440

    # 1. 定义目标变量 'r850'
    r850_channel_index = variables.index('r850')
    
    # 2. 获取用于反归一化的均值和标准差
    r850_mean = means[r850_channel_index]
    r850_std = stds[r850_channel_index]


    # 3. 获取数据文件和初始时间步
    data_file, prediction_length = get_data_for_date(year, month, day)
    
    # 4. 预处理初始条件
    initial_condition = preprocess(data_file)
    
    # 5. 执行推理
    normalized_predictions = inference(initial_condition, model, prediction_length, r850_channel_index, lat_idx, lon_idx)

    # 6. 后处理：反归一化并整理输出
    normalized_r850_list = normalized_predictions
    humidity_percent_list = []

    if not normalized_r850_list:
        return []

    for norm_rh in normalized_r850_list:
        # a. 反归一化
        # ERA5数据集中，相对湿度是以0-100的数值存储的。
        humidity_value = (norm_rh * r850_std) + r850_mean
        
        # b. 确保值在物理上合理的0-100范围内
        humidity_value = max(0.0, min(100.0, humidity_value))
        
        humidity_percent_list.append(humidity_value)
        
    return humidity_percent_list

@mcp.tool()
def inference_wind_info(year: int, month: int, day: int, city_name: str) -> List[Dict[str, float]]:
    """
    获取指定城市在指定日期的10米风速和风向预报。一次只能获取一天。
    风速和风向以6小时为间隔。

    Args:
        year (int): 年份
        month (int): 月份
        day (int): 日
        city_name (str): 城市名称

    Returns:
        List[Dict[str, float]]: 一个包含4个字典的列表。
                                每个字典代表一个时间点(00, 06, 12, 18时)，
                                包含 'speed' (单位: m/s) 和 'direction' (单位: 度) 两个键。
                                风向以正北为0度，顺时针方向。
    """
    try:
        lat, lon = get_city_coordinates(city_name)
        if lat is None or lon is None:
            return []
    except Exception as e:
        print(f"获取坐标时出错: {e}")
        return []

    lat_idx = int((90 - lat) / 0.25)
    lon_idx = int(lon / 0.25) % 1440

    # 定义我们需要的风速变量及其通道索引
    u10_channel_index = variables.index('u10')
    v10_channel_index = variables.index('v10')

    # 获取用于反归一化的均值和标准差
    u10_mean = means[u10_channel_index]
    u10_std = stds[u10_channel_index]
    v10_mean = means[v10_channel_index]
    v10_std = stds[v10_channel_index]

    # 1. 获取数据文件和初始时间步
    data_file, prediction_length = get_data_for_date(year, month, day)
    
    # 2. 预处理初始条件
    initial_condition = preprocess(data_file)
    u10_idx=variables.index('u10')
    v10_idx=variables.index('v10')
    

    # 4. 推理和后处理：反归一化并计算风速和风向
    normalized_u10_list = inference(initial_condition, model, prediction_length, u10_idx, lat_idx, lon_idx)
    normalized_v10_list = inference(initial_condition, model, prediction_length, v10_idx, lat_idx, lon_idx)
    
    wind_info_list = []
    # 确保我们同时得到了u和v分量
    if not normalized_u10_list or not normalized_v10_list:
        return []

    for norm_u, norm_v in zip(normalized_u10_list, normalized_v10_list):
        # a. 反归一化
        u10 = (norm_u * u10_std) + u10_mean
        v10 = (norm_v * v10_std) + v10_mean
        
        # b. 计算标量风速 (勾股定理)
        speed = np.sqrt(u10**2 + v10**2)
        
        # c. 计算风向
        # 使用 arctan2(u, v) 可以得到从正北方向顺时针测量的风向角（弧度）
        # 结果范围在 -pi 到 pi 之间。我们需要转换到 0-360 度。
        # 注意：在气象学中，风矢量(u,v)指向风去的方向，而风向是风来的方向。
        # 因此，要计算风来的方向，需要对矢量取反，即使用 arctan2(-u, -v)。
        direction_rad = np.arctan2(-u10, -v10)
        direction_deg = np.degrees(direction_rad)
        
        # 将角度从 (-180, 180] 转换到 [0, 360)
        if direction_deg < 0:
            direction_deg += 360
            
        wind_info_list.append({
            'speed': speed,
            'direction': direction_deg
        })
        
    return wind_info_list

@mcp.tool()
def get_today()  -> str:
    """
    获取今天日期
    
    Returns:
        str: 当前日期，格式为"年-月-日"
    """
    return '2025-06-20'

@mcp.tool()
def inference_t2m(year :int, month :int, day :int,city_name :str) -> List[float]:
    """
    获取指定城市在指定日期的气温预报。一次只能获取一天。
    气温以6小时为间隔。
    
    Args:
        year (int): 年份
        month (int): 月份
        day (int): 天数
        city_name (str): 城市名称
    
    Returns:
        List[float]: 当前地表温度/气温信息，以6hr为间隔，单位：摄氏度
    """
    try:
        lat, lon = get_city_coordinates(city_name)
        if lat is None or lon is None:
            return f"无法获取'{city_name}'的坐标。"
    except Exception as e:
        return f"获取坐标时出错: {e}"

    lat_idx = int((90 - lat) / 0.25)
    lon_idx = int(lon / 0.25) % 1440
    
    t2m_channel_index = variables.index('t2m')
    t2m_mean = means[t2m_channel_index]
    t2m_std = stds[t2m_channel_index]
    
    prediction_length = 4  # 预测4个时间步 (00, 06, 12, 18点)

    # 1. 获取数据文件和初始时间步
    data_file, prediction_length = get_data_for_date(year, month, day)
    
    # 2. 预处理初始条件 (只加载一个时间点)
    initial_condition = preprocess(data_file)
    
    # 3. 执行正确的自回归推理
    # 注意：inference现在返回的是归一化后的温度列表
    normalized_temps_list = inference(initial_condition, model, prediction_length, t2m_channel_index, lat_idx, lon_idx)

    # 4. 对结果进行反归一化和单位转换
    temperature_celsius_list = []
    for normalized_temp in normalized_temps_list:
        kelvin_temp = (normalized_temp * t2m_std) + t2m_mean
        celsius_temp = kelvin_temp - 273.15
        temperature_celsius_list.append(celsius_temp)
        
    return temperature_celsius_list
@mcp.tool()
def get_city_coordinates(city_name: str):
    """
    使用 geopy 和 Nominatim 服务获取指定城市的经纬度信息。

    Args:
        city_name (str): 城市的名称。为了提高准确性，可以包含省份/州和国家，
                         例如 "旧金山, 加利福尼亚州, 美国" 或 "西安市, 陕西省"。

    Returns:
        dict: 一个包含详细信息的字典，包括 'latitude', 'longitude', 'address'。
              如果找不到城市或服务不可用，则返回 None。
    """
    # 1. 创建一个地理编码器实例
    #    `user_agent` 是一个必需参数，你需要为你的应用指定一个唯一的名称。
    #    这是 Nominatim 的使用政策，以便他们可以识别请求来源。
    time.sleep(random.uniform(0.5, 1.5))
    geolocator = Nominatim(user_agent=random.choice(user_agent))

    try:
        # 2. 调用 geocode 方法进行地理编码
        #    timeout参数设置超时时间（秒），防止程序因网络问题卡住。
        location = geolocator.geocode(city_name, timeout=10)

        # 3. 检查是否成功找到了位置
        if location:
            # 成功找到，返回一个包含所需信息的字典
            return location.latitude, location.longitude
        else:
            # 未找到指定城市
            print(f"错误：无法找到城市 '{city_name}'。请检查拼写或尝试更详细的名称。")
            return None

    except GeocoderTimedOut:
        print("错误：地理编码服务超时。请检查您的网络连接或稍后再试。")
        return None
    except GeocoderUnavailable:
        print("错误：地理编码服务当前不可用。请稍后再试。")
        return None
    except Exception as e:
        print(f"发生未知错误: {e}")
        return None

# 仅测试
def load_image(path: str) -> str:
    """
    生成一个Logo图片，将其保存为文件，并返回该文件的绝对路径。
    这个函数不负责向用户展示图片。
    
    Returns:
        str:  图片被保存到的完整路径。主程序（MCP）需要用这个路径来向用户展示图片。
    """
    # FastMCP会处理读取和格式检测
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
    plt.show()
    return "D:/example.png"

@mcp.tool()
def draw(year :int, month :int, day :int, varname: str) -> str:
    """
    生成一个某日特定气象变量可视化图片，将其保存为文件，并返回该文件的绝对路径。
    这个函数不负责向用户展示图片。
    
    Args:
        year (int): 年份
        month (int): 月份
        day (int): 日期
        varname (str): 待可视化的变量名，你需要将用户提供的变量名转换成最接近的气象变量名。可选值为：'u10','v10', 't2m', 'sp', 'msl', 't850', 'u1000', 'v1000', 'z1000', 'u850', 'v850', 'z850', 'u500', 'v500', 'z500', 't500', 'z50' , 'r500', 'r850', 'tcwv'
        
    Returns:
        str:  图片被保存到的完整路径。主程序（MCP）需要用这个路径来向用户展示图片。
    """
    
    # 1. 获取数据文件和初始时间步
    data_file, prediction_length = get_data_for_date(year, month, day)
    
    # 2. 预处理初始条件 (只加载一个时间点)
    initial_condition = preprocess(data_file)
    
    # 3. 执行正确的自回归推理
    # 注意：inference现在返回的是归一化后的温度列表
    normalized_temps_list = inference_global(initial_condition, model, prediction_length)
    
    
    channel_idx=variables.index(varname)
    
    show_hotmap(normalized_temps_list,channel_idx)

    return "D:/example.png"

def show_hotmap(prediction,channel_idx):
    # 反归一化
    mean = means[channel_idx]
    std = stds[channel_idx]
    
    real_data = prediction[0, channel_idx] * std + mean  # 反标准化

    # 绘图
    fig, ax = plt.subplots(nrows=1, ncols=1)
    im = ax.imshow(real_data, cmap="bwr")
    ax.set_title("FourCastNet Prediction")
    
    # 添加颜色条
    cbar = fig.colorbar(im, ax=ax)

    # plt.savefig("D:/example.png")
    plt.show()

if __name__ == "__main__":
    print("Starting MCP server...")
    mcp.run(transport='stdio')
