from utility.DBUtility import DBUtility
from multiprocessing   import Queue, Process, Lock, Manager, Value, cpu_count
from datetime          import date, timedelta
import pandas as pd
import csv, os, sys, time

start = date(2018, 11, 1)
end = date(2019, 10, 5)

base_path = os.path.dirname(os.path.abspath(__file__))

WORKERS = cpu_count()

def Download(dt_queue, writelock, dbutil, finish, res):
    while True:
        try :
            dt = dt_queue.get()
        except Exception :
            break

        articles = dbutil.GetArticles({"time": str(dt), "state": "cn"})

        tmp = list()

        for article in articles:
            tmp.append({
                "account": article["account"],
                "date": str(dt),
                "official": article["official"],
                "censor": article["censor"],
                "license": article["license"],
                "forprofit": article["forprofit"]
            })

        writelock.acquire()

        res.extend(tmp)

        finish.value += 1

        writelock.release()

if __name__ == '__main__':
    writelock = Lock()
    dt_queue = Queue()
    manager = Manager()
    finish = Value("i", 0)
    res = manager.list()
    now = start
    dbutil = DBUtility()

    while True:
        if now > end:
            break
        dt_queue.put(str(now))
        now += timedelta(days = 1)


    for _ in range(WORKERS):
        t = Process(target = Download, args = (dt_queue, writelock, dbutil, finish, res))
        t.daemon = True
        t.start()

    total = dt_queue.qsize()

    sys.stdout.write("Downloading... 0%")

    while True :
        percent = round((total - dt_queue.qsize())/total*100, 2)
        sys.stdout.write('\r')
        sys.stdout.write("Downloading... {}%".format(percent))
        sys.stdout.flush()
        time.sleep(0.1)
        if finish.value == total and percent >= 100:
            break

    df = pd.DataFrame(list(res))
    df.to_csv(os.path.join(base_path, "Data.csv"))