# -*- coding: utf-8 -*-
import cv2
import numpy as np
import time
import os
from csv_process import read_csv

# HaarLike特徴抽出アルゴリズムのファイル
HAAR_FILE = "data/haarcascade_frontalface_alt.xml"


def trimming_face_image(image, face, size=(40,40)):
    """
    画像から顔を切り取り,リサイズした画像を返す
    """
    for x, y, w, h in face:
        # スライシングで顔の部分を切り取る
        face_image = image[y:y+h, x:x+w]
        # リサイズする 
        face_image = cv2.resize(face_image, size)
        return face_image

def save_face_image(dir_path, name, images_num=10, size=(40,40), fps=0.5):
    """
    dir_pathにname+count.pngの形式でimages_numの枚数分、sizeのサイズで顔画像を保存する
    """
    # カメラの開始
    cap = cv2.VideoCapture(0)
    count = 0
    # images_numの回数分繰り返し
    while count < images_num:
        # 映像データを読み込んでサイズ変更
        rst, stream = cap.read()
        stream = cv2.resize(stream, (320,240))
        
        # HaarLike特徴抽出アルゴリズムから分類器を作成
        cascade = cv2.CascadeClassifier(HAAR_FILE)
        # 実際に分類を行う
        face = cascade.detectMultiScale(stream)

        # 顔認識したのが一つだったら処理をする
        if len(face) == 1:
            # 顔画像を切り取る
            face_image = trimming_face_image(stream, face, size)
            # dir_path/name+count.pngの形式で保存 
            cv2.imwrite("{}/{}{}.png".format(dir_path, name, count), face_image)
            count += 1
            time.sleep(fps)

        # 認識した顔を赤い四角で囲う
        for x, y, w, h in face:
            cv2.rectangle(stream, (x,y), (x+w,y+h), (0,0,255), 1)

        # 画像をウインドウに表示
        cv2.imshow("img", stream)
        
        # 'q'を入力でアプリケーション終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    #終了処理
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 保存する枚数
    images_num = 20
    id = 5
    base_csv = "data/arasi.csv"
    data = read_csv(base_csv)
    name = data["name"][id]
    # name = "ninomiya"

    # 画像を保存するディレクトリ
    dir_name = "train_data_arasi"
    dir_path = dir_name + "/" + str(id)
    # csv_file = dir_name + ".csv"
    size = (40, 40)
    fps = 0.5
    try:
        os.makedirs(dir_path)
    except FileExistsError:
        pass
    # write_csv(csv_file, id, name)
    save_face_image(dir_path, name, images_num, size, fps)
