"""
天气查询工具模块
提供真实和模拟的天气查询功能
"""
import requests
from typing import Optional
from langchain_core.tools import tool


# 使用免费的 wttr.in API，不需要 API key
@tool
def get_weather(city: str) -> str:
    """
    查询指定城市的天气信息

    Args:
        city: 城市名称，例如 "北京", "上海", "New York"

    Returns:
        天气信息的字符串描述
    """
    try:
        # 使用 wttr.in 免费天气 API
        base_url = "https://wttr.in"
        # 使用格式化的 JSON 响应
        url = f"{base_url}/{city}?format=j1"

        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        # 解析天气数据
        current = data.get('current_condition', [{}])[0]
        area = data.get('nearest_area', [{}])[0]

        location = area.get('areaName', [{}])[0].get('value', city)
        temp = current.get('temp_C', 'N/A')
        feels_like = current.get('FeelsLikeC', 'N/A')
        humidity = current.get('humidity', 'N/A')
        weather_desc = current.get('weatherDesc', [{}])[0].get('value', '未知')
        wind_speed = current.get('windspeedKmph', 'N/A')

        result = f"""
📍 地点: {location}
🌡️ 温度: {temp}°C (体感 {feels_like}°C)
☁️ 天气: {weather_desc}
💧 湿度: {humidity}%
💨 风速: {wind_speed} km/h
"""
        return result.strip()

    except requests.RequestException as e:
        # 如果网络请求失败，返回模拟数据
        return _get_mock_weather(city)


def _get_mock_weather(city: str) -> str:
    """
    返回模拟的天气数据（当网络请求失败时使用）
    """
    mock_weather_data = {
        "北京": {"temp": 22, "weather": "晴朗", "humidity": 45, "wind": 12},
        "上海": {"temp": 25, "weather": "多云", "humidity": 65, "wind": 15},
        "广州": {"temp": 28, "weather": "阵雨", "humidity": 80, "wind": 10},
        "深圳": {"temp": 27, "weather": "阴天", "humidity": 75, "wind": 8},
        "New York": {"temp": 18, "weather": "Cloudy", "humidity": 55, "wind": 20},
        "London": {"temp": 15, "weather": "Rainy", "humidity": 70, "wind": 18},
    }

    if city in mock_weather_data:
        data = mock_weather_data[city]
        return f"""
📍 地点: {city}
🌡️ 温度: {data['temp']}°C (模拟数据)
☁️ 天气: {data['weather']}
💧 湿度: {data['humidity']}%
💨 风速: {data['wind']} km/h

注意: 当前使用模拟数据，请检查网络连接
""".strip()
    else:
        return f"""
📍 地点: {city}
🌡️ 温度: 20°C
☁️ 天气: 晴朗
💧 湿度: 50%
💨 风速: 10 km/h

注意: 城市 '{city}' 暂无详细数据，以上为模拟数据
""".strip()


@tool
def get_forecast(city: str, days: int = 3) -> str:
    """
    获取指定城市的天气预报

    Args:
        city: 城市名称
        days: 预报天数 (1-3)

    Returns:
        天气预报信息
    """
    try:
        base_url = "https://wttr.in"
        url = f"{base_url}/{city}?format=j1"

        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        weather_list = data.get('weather', [])
        days = min(days, len(weather_list), 3)

        result = f"📍 {city} 未来 {days} 天天气预报:\n\n"

        for i in range(days):
            day_data = weather_list[i]
            date = day_data.get('date', '未知日期')
            max_temp = day_data.get('maxtempC', 'N/A')
            min_temp = day_data.get('mintempC', 'N/A')
            weather_desc = day_data.get('hourly', [{}])[0].get('weatherDesc', [{}])[0].get('value', '未知')

            result += f"📅 {date}\n"
            result += f"  温度: {min_temp}°C - {max_temp}°C\n"
            result += f"  天气: {weather_desc}\n\n"

        return result.strip()

    except Exception as e:
        return f"获取天气预报失败: {str(e)}\n提示: 请检查城市名称是否正确或网络连接"


# 导出工具列表
tools = [get_weather, get_forecast]
