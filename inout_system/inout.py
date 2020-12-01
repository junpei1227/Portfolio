from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os
from csv_process import read_csv, write_csv
try:
    from amg8833 import  get_temperature_max, get_temperature_8x8
except ImportError:
    pass

def check_second_difference(check_time,second=120):
    """
    check_timeからsecond秒たったらTrue
    """
    now_time = pd.Timestamp.now()
    delta = now_time - pd.to_datetime(check_time)
    if int(delta.seconds) >= second:
        return True
    else:
        return False

def check_late_early(check_time, start="0930", end="1530"):
    """
    check_timeがstartとendの間か判定し、遅刻：早退の時間を分で辞書型で返す
    """
    # pandasのタイムスタンプ型に変換 
    now_time = pd.Timestamp.now()
    check_time = pd.to_datetime(check_time)
    start = pd.to_datetime(now_time.strftime("%Y%m%d") + start)
    end = pd.to_datetime(now_time.strftime("%Y%m%d") + end)

    if start < check_time < end:
        delta = end - check_time
        # 遅刻時間
        late = int((end - start - delta).seconds/60)
        # 早退時間
        early = int(delta.seconds/60) 

        late_early = {"late":late, "early":early}
        return late_early

def inout_pd_data(id, data):
    """
    idの人の入退室データをpandasの形式で書き込む
    """
    now_time = pd.Timestamp.now()
    now_time = now_time.strftime("%Y-%m-%d %H:%M")
    # 入室時の処理
    if pd.isnull(data["in"][id]):
        data["in"][id] = now_time
        data["status"][id] = "◎"
        # 体温の取得
        try:
            sensor_data = get_temperature_max()
            print(sensor_data)
            data["temperature"][id] = sensor_data
        except Exception:
            pass
        # 遅刻時の処理
        if check_late_early(data["in"][id]) != None:
            data["late"][id] = check_late_early(data["in"][id])["late"]
            data["status"][id] = "△"
    # 退室時の処理
    elif pd.isnull(data["out"][id]):
        if check_second_difference(data["in"][id]):
            data["out"][id] = now_time
            if pd.isnull(data["late"][id]):
                data["status"][id] = "○"
            # 早退時の処理
            if check_late_early(data["out"][id]) != None:
                data["early"][id] = check_late_early(data["out"][id])["early"]
                data["status"][id] = "▽"
            
    # 中抜け時の処理
    elif pd.isnull(data["dropin"][id]):
        if check_second_difference(data["out"][id]):
            data["dropin"][id] = now_time
            data["status"][id] = "□"

    # 中抜けの人の退室時の処理
    elif pd.isnull(data["dropout"][id]):
        if check_second_difference(data["dropin"][id]):
            data["dropout"][id] = data["out"][id]
            data["out"][id] = now_time
            droptime = pd.to_datetime(data["dropin"][id]) - pd.to_datetime(data["dropout"][id])
            droptime = int(droptime.seconds/60)
            data["droptime"][id] = droptime
            if check_late_early(data["out"][id]) != None:
                data["early"][id] = check_late_early(data["out"][id])["early"]
    return data



def inout_to_csv(id, csv_file):
    """
    idの人の入退室データをcsv_fileに書き込む
    """
    data = read_csv(csv_file)
    data = inout_pd_data(id, data)
    print(data)
    write_csv(data, csv_file)

            
                

if __name__ == "__main__":
    now = pd.Timestamp.now()
    today_csv = now.strftime("%Y-%m-%d") + ".csv"
    base_csv = "data/arasi.csv"
    if os.path.exists(today_csv) == False:
        today_data = read_csv(base_csv)
        write_csv(today_data, today_csv)
    # today_data = read_csv(today_csv)
    # id = 3
    # today_data = inout_pd_data(id, today_data)
    # write_csv(today_data, today_csv)
    id = 2
    inout_to_csv(id, today_csv)

