import requests
import json
import re
def get_weather(airport4, start_time, end_time): #根据机场四字码和时间段获取气象数据
    url = "http://www.baidu.com/getflightmetarinfo/"+airport4+"/"+start_time+"/"+end_time
    response = requests.get(url=url, timeout =30)
    response.encoding = "GBK"
    result = json.loads(response.text)[0]
    status = result["resultcode"]
    if status=='0': #如果请求成功,有数据返回
        weather_content = result["resultdata"][-1] #筛选最后一个
        content = weather_content["content"] #气象主体
        # print(content)
        airport4 = re.findall(r' [A-Z]{4}', content)[0].replace(' ', '') #机场四字码
        ob_time = re.findall(r'\d+Z', content)[0].replace("Z", "")  #观测时间
        wind = re.search(r'V{0,1}R{0,1}B{0,1}\d+[A-Z]{0,1}\d+(MPS|KT)', content).group() #风
        wind_speed = re.search(r'\d{2}(MPS|KT)', wind).group().replace('MPS', '').replace('KT', '') #风速
        wind_towards = wind[0:3] #风向
        try:
             visib = re.findall(r'CAVOK| \d{4} ', content)[0].replace(' ', '') #能见度
        except:
            visib = ''
        rvr = ' '.join(s1 for s1 in re.findall(r'R\d{2}[A-Z]{0,1}\/.+', content))# 跑道能见度
        try:
            nowweather = re.findall(r'-?\+?[A-Z]{6}|-?\+?[A-Z]{2,4}', content)[0].replace(' ', '') #现在天气
            if nowweather == 'META':
                nowweather = ''
            else:
                nowweather = nowweather
        except:
            nowweather = ''
        cloud = ' '.join(s1 for s1 in re.findall(r'VV\d{3}|[A-Z]{3}\d{3}[A-Z]{2,3}|NSC|NCD|SKC|[A-Z]{3}\d{3}', content)) #云层
        temper = re.findall(r' M{0,1}\d.\/M{0,1}\d. ', content)[0].split("/")[0].replace(' ', '') # 气温
        dew = re.findall(r' M{0,1}\d.\/M{0,1}\d. ', content)[0].split("/")[1].replace(' ', '') # 露点
        qnh = re.search(r'Q\d{4}', content).group().replace("Q", "") #修正海压
        try:
            trend = re.findall(r'NOSIG=.+', content)[0].replace('NOSIG= ', '') #未来趋势
        except:
            trend = ''
        weather = [ob_time, wind_speed, wind_towards, visib, rvr, nowweather, cloud, temper, dew, qnh, trend, content]
        # print(weather)
        return weather
get_weather('ZSNJ','2021-08-30 04:55:00','2021-12-30 04:55:00')