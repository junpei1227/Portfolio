# -*- coding: utf-8 -*-
import cv2
import numpy as np
# try:
# from sklearn.externals import joblib
# except ImportError:
import joblib
import pandas as pd
import os
import re
from pyzbar.pyzbar import decode
import inout
from csv_process import read_csv, write_csv


def qr_inout_csv(csv_file,stream_size=(320,240), font=cv2.FONT_HERSHEY_SIMPLEX):
    """
    grコードを読み込み、その人の入退室データを書き込む
    """
    data = read_csv(csv_file)
    cap = cv2.VideoCapture(0)
    while True: 
        # 映像データを読み込んでサイズ変更
        rst, stream = cap.read()
        stream = cv2.resize(stream, stream_size)
        
        # streamからqrを検出する
        decoded_objs = decode(stream)
        if decoded_objs != []:
            for obj in decoded_objs:
                print('Type: ', obj)
                str_dec_obj = obj.data.decode('utf-8', 'ignore')
                # 文字列から数値を抽出
                id = int(re.sub("\\D","", str_dec_obj))
                # csvにidの人の入退室時間をpandasの形式で書き込む
                inout.inout_to_csv(id, csv_file)

                print('QR code: {}'.format(str_dec_obj))
                x, y, w, h = obj.rect
                # qrコードを四角で囲う
                cv2.rectangle(stream, (x,y), (x+w,y+h), (0,0,255), 1)
                # 文字列を表示
                cv2.putText(stream, str_dec_obj,(x,y-10), font, 1, (0,255,0), 3, cv2.LINE_AA)

        # 画像をウインドウに表示
        cv2.imshow("img", stream)
        
        # 'q'を入力でアプリケーション終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    #終了処理
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    now = pd.Timestamp.now()
    today_csv = now.strftime("%Y-%m-%d") + ".csv"
    base_csv = "data/train_data_arasi.csv"
    if os.path.exists(today_csv) == False:
        today_data = read_csv(base_csv)
        write_csv(today_data, today_csv)

    qr_inout_csv(today_csv)