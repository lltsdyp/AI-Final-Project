import cdsapi
import os
import time

def get_user_date_input():
    """
    获取用户输入的年份、月份、开始和结束日期，并进行基本验证。
    """
    try:
        year = int(input("请输入年份 (例如, 2021): "))
        month = int(input("请输入月份 (1-12): "))
        start_day = int(input("请输入开始日期 (1-31): "))
        end_day = int(input("请输入结束日期 (1-31): "))
        basedir=input("请输入保存数据的文件夹路径: ")

        if not (1 <= month <= 12 and 1 <= start_day <= 31 and 1 <= end_day <= 31 and start_day <= end_day):
            print("错误：日期范围无效，请重新输入。")
            return None, None, None,None
        
        # 生成日期列表
        days = [str(d) for d in range(start_day, end_day + 1)]
        
        # 将月份格式化为两位数，例如 9 -> '09'
        month_str = f"{month:02d}"
        
        return str(year), month_str, days,basedir
    except ValueError:
        print("错误：输入无效，请输入数字。")
        return None, None, None,None

def download_surface_data(client, year, month, days,basedir):
    """
    下载ERA5地面层数据 (sfc)。
    """
    output_filename = f"era5_{year}_{month}_sfc.nc"
    print(f"\n准备下载地面数据 (sfc) 到文件: {output_filename}...")
    
    try:
        client.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': [
                    '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature',
                    'mean_sea_level_pressure', 'surface_pressure', 'total_column_water_vapour',
                ],
                'year': year,
                'month': month,
                'day': days,
                'time': ['00:00', '06:00', '12:00', '18:00'],
            },
            os.path.join(basedir,output_filename)
        )
        print(f"成功: 地面数据已保存到 {os.path.abspath(output_filename)}")
    except Exception as e:
        print(f"错误: 下载地面数据失败。原因: {e}")

def download_pressure_level_data(client, year, month, days,basedir):
    """
    下载ERA5多气压层数据 (pl)。
    """
    output_filename = f"era5_{year}_{month}_pl.nc"
    print(f"\n准备下载多气压层数据 (pl) 到文件: {output_filename}...")

    try:
        client.retrieve(
            'reanalysis-era5-pressure-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': [
                    'geopotential', 'relative_humidity', 'temperature',
                    'u_component_of_wind', 'v_component_of_wind',
                ],
                'pressure_level': ['50', '500', '850', '1000'],
                'year': year,
                'month': month,
                'day': days,
                'time': ['00:00', '06:00', '12:00', '18:00'],
            },
            os.path.join(basedir,output_filename)
        )
        print(f"成功: 气压层数据已保存到 {os.path.abspath(output_filename)}")
    except Exception as e:
        print(f"错误: 下载气压层数据失败。原因: {e}")

def main():
    """
    主函数，协调用户输入和数据下载流程。
    """
    print("ERA5 数据下载脚本")
    print("=" * 20)
    
    # 获取用户输入
    year, month, days, basedir = get_user_date_input()
    if basedir is None:
        basedir="."
    
    if not all((year, month, days)):
        return # 如果输入无效则退出
        
    try:
        # 初始化CDS API客户端
        c = cdsapi.Client()

        # 下载两种类型的数据
        download_surface_data(c, year, month, days,basedir)
        time.sleep(60)
        download_pressure_level_data(c, year, month, days,basedir)
        
        print("\n所有下载任务已完成。")

    except Exception as e:
        print(f"\n发生严重错误: {e}")
        print("请确保您的 'cdsapi' 库已安装，并且 `~/.cdsapirc` 文件已正确配置。")

if __name__ == "__main__":
    main()