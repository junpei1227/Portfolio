# -*- coding: utf-8 -*-

import time
import numpy as np
import busio
import board

import adafruit_amg88xx
def get_temperature_max():
    """"""
    # I2Cバスの初期化
    i2c_bus = busio.I2C(board.SCL, board.SDA)

    # センサーの初期化
    # アドレスを68に指定
    sensor = adafruit_amg88xx.AMG88XX(i2c_bus, addr=0x68)

    # センサーの初期化待ち
    time.sleep(.1)

    # 一番温度の高いところを返す
    sensor_data = np.array(sensor.pixels)
    sensor_data = np.amax(sensor_data)
    return sensor_data


def get_temperature_8x8():
    """"""
    # I2Cバスの初期化
    i2c_bus = busio.I2C(board.SCL, board.SDA)

    # センサーの初期化
    # アドレスを68に指定
    sensor = adafruit_amg88xx.AMG88XX(i2c_bus, addr=0x68)

    # センサーの初期化待ち
    time.sleep(.1)

    # 8x8の表示
    sensor_data = np.array(sensor.pixels)
    return sensor_data

if __name__ == "__main__":
    sensor_data = get_temperature_8x8()
    print(sensor_data)
    print(np.amax(sensor_data))
    print(dir(sensor_data))