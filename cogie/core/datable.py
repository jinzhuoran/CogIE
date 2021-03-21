"""
@Author: jinzhuan
@File: datable.py
@Desc: 
"""
from cogie.utils import save_json, load_json
import prettytable as pt


class DataTable:
    def __init__(self, headers=None):
        self.datas = {}
        self.not2torch = set()
        if not (isinstance(headers, list) or headers is None):
            raise TypeError("Headers must be a list!")
        if headers:
            self.headers = headers
        else:
            self.headers = []

    def __len__(self):
        length = 0
        for header in self.headers:
            if length == 0:
                length = len(self.datas[header])
            else:
                if length != len(self.datas[header]):
                    raise ValueError("Different data lengths cause errors!")
        return length

    def __call__(self, *args, **kwargs):
        if len(args) == 1:
            if isinstance(args[0], list):
                self.add_headers(args[0])
            else:
                self.add_header(args[0])
        elif len(args) == 2:
            self.add_data(args[0], args[1])

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.get_row(index)
        elif isinstance(index, str):
            return self.get_col(index)
        else:
            raise AttributeError("Please input int or str")

    def add_header(self, header):
        if header not in self.headers:
            self.headers.append(header)
            self.datas[header] = []
        else:
            pass

    def add_headers(self, headers):
        for header in headers:
            self.add_header(header)
        return

    def add_data(self, header, data):
        if header not in self.headers:
            self.add_header(header)
        self.datas[header].append(data)

    def save_table(self, path):
        save_json(self.datas, path)

    @staticmethod
    def load_table(path):
        datable = DataTable()
        datable.datas = load_json(path)
        datable.headers = list(datable.datas.keys())
        return datable

    def get_row(self, index):
        row = []
        for head in self.headers:
            row.append(self.datas[head][index])
        return row

    def get_col(self, header):
        if header in self.headers:
            return self.datas[header]
        else:
            raise ValueError("Illegal header name!")

    def print_table(self, row_num):
        tb = pt.PrettyTable()
        tb.field_names = ["index"] + self.headers
        for i in range(row_num if row_num <= len(self) else len(self)):
            row = [i]
            for head in self.headers:
                row.append(self.datas[head][i])
            tb.add_row(row)
        print(tb)

    def split(self, train, dev, test):
        train_datable = DataTable(self.headers)
        dev_datable = DataTable(self.headers)
        test_datable = DataTable(self.headers)
        train_prop = int(float(train) / (train + dev + test) * self.__len__())
        dev_prop = int(float(dev) / (train + dev + test) * self.__len__())
        for header in self.headers:
            train_datable.datas[header] = self.datas[header][:train_prop]
            dev_datable.datas[header] = self.datas[header][train_prop:train_prop + dev_prop]
            test_datable.datas[header] = self.datas[header][train_prop + dev_prop:]
        return train_datable, dev_datable, test_datable

    def add_not2torch(self, header):
        self.not2torch.add(header)
