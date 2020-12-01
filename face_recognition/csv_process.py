import pandas as pd
import csv


def read_csv(csv_file):
    header = pd.read_csv( csv_file, index_col=0, names=["id", "name","temperature", "status", "in", "dropout", "dropin", "out", "late", "early", "droptime"])
    return header


def write_csv(data, csv_file):
    data.to_csv(csv_file, header=False)


if __name__ == "__main__":
    csv_file = "data/arasi.csv"
    data = read_csv(csv_file)
    write_csv(data,"today.csv")