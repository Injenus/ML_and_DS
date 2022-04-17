import random
from datetime import datetime
import csv
import multiprocessing as mp
from multiprocessing import Process

# def read2list(file):
#     # открываем файл в режиме чтения utf-8
#     file = open(file, 'r', encoding='utf-8')
#     # читаем все строки и удаляем переводы строк
#     lines = file.readlines()
#     lines = [int(line.rstrip('\n')) for line in lines]
#     file.close()
#     return lines
def is_in(lst, n):
    return n in lst

if __name__ == '__main__':




    start_time = datetime.now()
    # arr = read2list('list.txt')
    arr = []
    with open('list.csv', newline='') as myFile:
        reader = csv.reader(myFile, delimiter=',', quoting=csv.QUOTE_NONE)
        for row in reader:
            arr.append(int(row[0]))
    print('Чтение списка:', datetime.now() - start_time)

    pr = mp.Process(target=is_in, args=(arr, 5))
    start_time = datetime.now()
    print(pr.start())
    #pr.join()
    print('Поиск в списке через процесс:', datetime.now() - start_time)

    start_time = datetime.now()
    print(is_in(arr, 5))
    print('Поиск в списке:', datetime.now() - start_time)

    start_time = datetime.now()
    set_arr = set(arr)
    print('Создание множества:', datetime.now() - start_time)

    start_time = datetime.now()
    print(is_in(set_arr, 5))
    print('Поиск в множестве:', datetime.now() - start_time)

    # start_time = datetime.now()
    # tuple_arr = tuple(arr)
    # print('Создание кортежа:', datetime.now() - start_time)
    #
    # start_time = datetime.now()
    # print(is_in(tuple_arr, 5))
    # print('Поиск в кортеже:', datetime.now() - start_time)

    # start_time = datetime.now()
    # dct = tuple(arr)
    # print('Создание кортежа:', datetime.now() - start_time)
    #
    # start_time = datetime.now()
    # print(is_in(dct, 5))
    # print('Поиск в кортеже:', datetime.now() - start_time)
