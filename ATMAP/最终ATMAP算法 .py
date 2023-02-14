import re

# CON['MetarString'][i]='METAR ZSNJ 010000Z 21001MPS 8000 NSC M06/M07 Q1032 NOSIG='
import pandas as pd

CON = pd.read_csv('METAR3.csv')
# print(CON['MetarString'][i])
point1 = []
for i in range(CON.shape[0]):

    # CON['MetarString'][i] = 'METAR ZBAA 190800Z 13007G12MPS 090V170 9999 -TSRA BKN040CB 29/24 Q1004 NOSIG='
    # str_list = str(text.split())
    airport4 = re.findall(r' [A-Z]{4}', CON['MetarString'][i])[0].replace(' ', '')  # 机场四字码
    ob_time = re.findall(r'\d+Z', CON['MetarString'][i])[0].replace("Z", "")  # 观测时间
    wind = re.search(r'V{0,1}R{0,1}B{0,1}\d+[A-Z]{0,1}\d+(MPS|KT)', CON['MetarString'][i]).group()  # 风
    wind_speed = re.search(r'\d{2}(MPS|KT)', wind).group().replace('MPS', '').replace('KT', '')  # 风速
    wind_towards = wind[0:3]  # 风向
    visib = re.findall(r'CAVOK| \d{4} ', CON['MetarString'][i])[0].replace(' ', '')  # 能见度
    try:
        visib = re.findall(r'CAVOK| \d{4} ', CON['MetarString'][i])[0].replace(' ', '')  # 能见度
    except:
        visib = ''
    rvr = ' '.join(s1 for s1 in re.findall(r'R\d{2}[A-Z]{0,1}\/.+', CON['MetarString'][i]))  # 跑道能见度
    try:
        nowcon = re.findall(r'-?\+?[A-Z]{6}|-?\+?[A-Z]{2,4}', CON['MetarString'][i])[0].replace(' ', '')  # 现在天气
        if nowcon == 'META':
            nowcon = ''
        else:
            nowcon = nowcon
    except:
        nowcon = ''
    cloud = ' '.join(s1 for s1 in re.findall(r'VV\d{3}|[A-Z]{3}\d{3}[A-Z]{2,3}|NSC|NCD|SKC|[A-Z]{3}\d{3}',
                                             CON['MetarString'][i]))  # 云层
    temper = re.findall(r' M{0,1}\d.\/M{0,1}\d. ', CON['MetarString'][i])[0].split("/")[0].replace(' ', '').replace('M',
                                                                                                                    '-')  # 气温
    temper1 = int(temper)
    dew = re.findall(r' M{0,1}\d.\/M{0,1}\d. ', CON['MetarString'][i])[0].split("/")[1].replace(' ', '').replace('M',
                                                                                                                 '-')  # 露点
    dew1 = int(dew)
    qnh = re.search(r'Q\d{4}', CON['MetarString'][i]).group().replace("Q", "")  # 修正海压
    try:
        trend = re.findall(r'NOSIG.+.*|TEMPO.*|BECMG.*', CON['MetarString'][i])[0].replace('NOSIG ', '')  # 未来趋势
    except:
        trend = ''
    weather = str([wind_speed, visib, rvr, nowcon, cloud, temper, dew, qnh, trend])
    # print(CON['MetarString'][i])
    points = []
    point = 0
    # 能见度
    if visib == 'CAVOK':
        visib = 10000
    else:
        visib = int(visib)
    if 3200 < visib <= 5000:
        point = point + 2
        points.append(point)
    elif 1600 < visib <= 3200:
        point = point + 4
        points.append(point)
    elif visib <= 1600:
        point = point + 5
        points.append(point)
    # 风速
    wind_speed = int(wind_speed)
    if 7.7 <= wind_speed < 10.28:
        point = point + 1
        points.append(point)
    if 1028 <= wind_speed < 15.42:
        point = point + 2
        points.append(point)
    if 15.42 <= wind_speed :
        point = point + 4
        points.append(point)
    # 降水
    try:
        rain2 = re.findall(r'RA|UP|DZ|IC|-RA.*', CON['MetarString'][i])[0]
        if len(rain2) != 0:
            point = point + 1
            points.append(point)

    except:
        rain2 = 0
    try:
        rain_2 = re.findall(r'\+RA.*', CON['MetarString'][i])[0]
        if len(rain_2) != 0:
            point = point - 1
            points.append(point)
    except:
        rain_2 = 0
    try:
        rain3 = re.findall(r'-SN|SG|\+RA|\+SH.*\+TSRA', CON['MetarString'][i])[0]
        if len(rain2) != 0:
            point = point + 2
            points.append(point)
    except:
        rain3 = 0
    try:
        rain4 = re.findall(r'FZ\SN\+SN', CON['MetarString'][i])[0]
        if len(rain3) != 0:
            point = point + 3
            points.append(point)
    except:
        rain4 = 0
    # 冰冻

    if -15 < temper1 <= 3 and temper1 - dew1 <= 3:
        point = point + 1
        points.append(point)
    try:
        freezing4 = re.findall(r'-SN|SG|\+RA|RASN|BR', CON['MetarString'][i])[0]
        if -15 < temper1 <= 3 and len(freezing4) != 0:
            point = point + 3
            points.append(point)
    except:
        freezing4 = 0
    try:
        freezing5 = re.findall(r'-SN|SG|\+RA|RASN|BR', CON['MetarString'][i])[0]
        if -15 < temper1 <= 3 and len(freezing5) != 0:
            point = point + 4
            points.append(point)
    except:
        freezing5 = 0

    # try:
    #     freezing5 = re.findall(r'-SN|SG|\+RA|RASN|BR', CON['MetarString'][i])[0]
    #    if -15<temper1<=3 and len(freezing5)!=0:
    #         point = point + 4
    #         points.append(point)
    # except:
    #         freezing5 = 0

    # 危险天气
    try:
        danger1 = re.findall(r'FEW[0-9]{3}CB', CON['MetarString'][i])[0]

        if len(danger1) != 0:
            point = point + 4
            points.append(point)

    except:
        danger1 = 0
    try:
        danger2 = re.findall(r'FEW[0-9]{3}TCU', CON['MetarString'][i])
        danger_2 = re.findall(r'-SH..', CON['MetarString'][i])
        danger_2_1 = re.findall(r'SH..|\+SH..', CON['MetarString'][i])
        if len(danger2) != 0:
            point = point + 3
            points.append(point)
            if len(danger_3) != 0:
                point = point + 1
                points.append(point)
            elif len(danger_3_1) != 0:
                point = point + 3
                points.append(point)
    except:
        danger2 = 0
    try:
        danger3 = re.findall(r'SCT[0-9]{3}CB', CON['MetarString'][i])
        danger_3 = re.findall(r'-SH..', CON['MetarString'][i])
        danger_3_1 = re.findall(r'SH..|\+SH..', CON['MetarString'][i])
        if len(danger3) != 0:
            point = point + 6
            points.append(point)
            if len(danger_3) != 0:
                point = point + 4
                points.append(point)
            elif len(danger_3_1) != 0:
                point = point + 9
                points.append(point)
    except:
        danger3 = 0
    try:
        danger4 = re.findall(r'SCT[0-9]{3}TCU', CON['MetarString'][i])
        danger_4 = re.findall(r'-SH..', CON['MetarString'][i])
        danger_4_1 = re.findall(r'SH..|\+SH..', CON['MetarString'][i])
        if len(danger4) != 0:
            point = point + 5
            points.append(point)
            if len(danger_4) != 0:
                point = point + 3
                points.append(point)
            elif len(danger_4_1) != 0:
                point = point + 7
                points.append(point)
    except:
        danger4 = 0

    try:
        danger5 = re.findall(r'BKN[0-9]{3}CB', CON['MetarString'][i])[0]
        danger_5 = re.findall(r'-SH..', CON['MetarString'][i])
        danger_5_1 = re.findall(r'SH..|\+SH..', CON['MetarString'][i])
        if len(danger5) != 0:
            point = point + 10
            points.append(point)
            if len(danger_5) != 0:
                point = point + 2
                points.append(point)
            elif len(danger_5_1) != 0:
                point = point + 10
                points.append(point)

    except:
        danger5 = 0
    try:
        danger6 = re.findall(r'BKN[0-9]{3}TCU', CON['MetarString'][i])[0]
        danger_6 = re.findall(r'-SH..', CON['MetarString'][i])
        danger_6_1 = re.findall(r'SH..|\+SH', CON['MetarString'][i])
        if len(danger6) != 0:
            point = point + 8
            points.append(point)
            if len(danger_6) != 0:
                point = point + 2
                points.append(point)
            elif len(danger_6_1) != 0:
                point = point + 5
                points.append(point)
    except:
        danger6 = 0
    try:
        danger7 = re.findall(r'OVC[0-9]{3}CB', CON['MetarString'][i])[0]
        danger_7 = re.findall(r'-SH..', CON['MetarString'][i])
        danger_7_1 = re.findall(r'SH..|\+SH', CON['MetarString'][i])
        if len(danger7) != 0:
            point = point + 12
            points.append(point)
            if len(danger_7) != 0:
                point = point + 6
                points.append(point)
            elif len(danger_7_1) != 0:
                point = point + 12
                points.append(point)

    except:
        danger7 = 0
    try:
        danger8 = re.findall(r'OVC[0-9]{3}TCU', CON['MetarString'][i])[0]
        danger_8 = re.findall(r'-SH..', CON['MetarString'][i])
        danger_8_1 = re.findall(r'SH..|\+SH', CON['MetarString'][i])
        if len(danger8) != 0:
            point = point + 10
            points.append(point)
            if len(danger_7) != 0:
                point = point + 2
                points.append(point)
            elif len(danger_7_1) != 0:
                point = point + 10
                points.append(point)
    except:
        danger8 = 0

    if len(points) == 0:
        points.append(0)
        # print(points)

    point1.append(points[-1])
# print(points[-1])
# temper = int(temper)
# dew = int (temper)
# a = temper -dew
result = pd.DataFrame(point1)
result.to_csv('补充ATMAP量化结果.csv')