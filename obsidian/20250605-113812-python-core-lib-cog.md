
### 内建库

机制
- [[#threading]] 异步
- [[#concurrent]] 并行(线程池)
- [[#asyncio]] 线程
- [[#typing]] 类型检查
- [[#importlib]] 动态导入模块
- [[#functools]] 函数工具
- [[#abc]] 抽象基类
- [[#atexit]] 程序退出时的回调事件

通用
- [[#re]] 正则匹配
- [[#glob]] 路径匹配
- [[#os]] 路径等相关
- [[#sys]] python 环境变量等
- [[#traceback]]
- [[#datetime]], [[#time]] 日期 时间
- [[#heapq]] 堆
- [[#itertools]] 迭代工具
- [[#bisect]] 二分

数据相关
- [[#zipfile]] 压缩文件
- [[#pickle]] python 特有 `.pkl` 文件
- [[#sqlite3]] `sqlite` 数据库
- [[#csv]] `.csv` 文件
- [[#tempfile]]


#### threading

[同步原语（Synchronization Primitives）_线程同步原语有哪些-CSDN博客](https://blog.csdn.net/u011019141/article/details/143889260)

- [[#^t-1]] Mutex Lock(互斥锁)
- [[#^t-2]] Semaphore(信号量)
- [[#^t-3]] Barrier(屏障)
- [[#^t-4]] Condition(条件变量)
- [[#^t-5]] Event(事件)

- read-write lock(读写锁)
- spin lock(自旋锁)

---
Mutex Lock(互斥锁) ^t-1

互斥锁 用于确保在同一段时间只有一个线程能访问某一资源

```python
import threading

lock = threading.Lock()

shared_resource = 0

import time, random
def f():
    global shared_resource
    name = threading.current_thread().name
    for j in range(3):
        j+=1
        print(f'{name}({j}): try to gain shared_resource...')
        with lock:
            shared_resource += 1
            print(f'{name}({j}): gain shared_resource!!!')
            time.sleep(random.random())

threadList = []
for i in range(3):
    th=threading.Thread(target=f)
    threadList.append(th)
    th.start()

for th in threadList:
    th.join()

print(shared_resource)
```


---
Semaphore(信号量) ^t-2

信号量 用于确保在同一段时间最多只有 n 个线程能访问某一资源
注：n=2 时表示 互斥锁
（比喻：旋转木马只有 n 个位置，所以最多运行 n 个人坐）

```python
import threading

semaphore = threading.Semaphore(value=3)

shared_resources = 0
workingThreadList = []

import time, random
def f():
    global shared_resources
    name = threading.current_thread().name
    with semaphore:
        workingThreadList.append(name)
        print(f'{name}: {workingThreadList}')
        time.sleep(random.random()*2)
        shared_resources += 1
        workingThreadList.remove(name)
        print(f'{name}(end): {workingThreadList}')


threadList = []
for i in range(7):
    th = threading.Thread(target=f)
    threadList.append(th)
    th.start()

for th in threadList:
    th.join()

print(shared_resources)
```


---
Barrier(屏障) ^t-3

线程等待的数量为 n 个(或 n 的倍数)时，这些线程结束等待，运行接下来的代码
（比喻：一个大门必须由 n 个人合力推开，而且这 n 个人打开并进入大门内侧后，大门自动关闭）

```python
import threading

# 注：若设置了 timeout，超时后会抛出 BrokenBarrierError
barrier = threading.Barrier(3, lambda: print('running barrier...'), timeout=2)

threadList = []

def f():
    name = threading.currentThread().name
    print(f'{name}: start')
    import time, random
    time.sleep(random.random()*2)
    print(f'{name}: wait in barrier ...')
    barrier.wait()
    print(f'{name}: end')

for i in range(9):
    th = threading.Thread(target=f)
    threadList.append(th)
    th.start()

for th in threadList:
    th.join()
```

其他方法：
```python
# 使所有 正在调用/即将调用 <Barrier>.wait() 的线程抛出 BrokenBarrierError
abort()
# 重置 <Barrier> 的计数，使所有 正在调用 <Barrier>.wait() 的线程抛出 BrokenBarrierError
reset()
```


---
Condition(条件变量) ^t-4

条件变量 的两个使用方法：
- 让 某个/多个 线程进行等待，当 线程被唤醒时 结束等待
- 让 某个/多个 线程进行等待某个条件，当 线程被唤醒时 并且 条件(一个回调函数的返回值)为 True 结束等待

例子1：
```python
import threading

condition = threading.Condition()

def f():
    name = threading.currentThread().name
    with condition:
        print(f'{name}: waiting...')
        condition.wait()
        print(f'{name}: end')

import time, random
def g():
    time.sleep(random.random()*2)
    with condition:
        condition.notify_all()
        # 或者
        # condition.notify(n=5)

for i in range(5):
    threading.Thread(target=f).start()
threading.Thread(target=g).start()

```

例子2：
```python
import threading
import time, random

condition = threading.Condition()

def f():
    name = threading.currentThread().name
    with condition:
        print(f'{name}: waiting...')
        condition.wait_for(lambda: random.random()*3<1)
        print(f'{name}: end')

def g():
    while True:
        time.sleep(0.5+random.random())
        with condition:
            condition.notify_all()


for i in range(5):
    threading.Thread(target=f).start()
threading.Thread(target=g).start()
```

---
Event(事件) ^t-5

线程调用 `<Event>.wait()` 后会一直等待，直到调用了 `<Event>.set()`，而且后续 `<Event>.wait()` 不会等待（除非 `<Event>.clear()`）

```python
import threading

event = threading.Event()

import time, random
def f():
    print('f: preparing resources...')
    time.sleep(1+random.random()*2)
    print('f: resources prepared!!!')
    event.set()


def g(i):
    print(f'g{i}: waiting resources')
    event.wait()
    print(f'g{i}: resources enough!!!')

threading.Thread(target=f).start()
for i in range(4):
    threading.Thread(target=g, args=[i]).start()
```

```python
# 将 flag 设为 True
set()
# 将 flag 设为 False
clear()
# 判断是否阻塞
is_set()

# 若 flag==False, 则一直等待
wait()
```

#### concurrent

---
例子：线程池

例子：线程池
```python
from concurrent.futures import ThreadPoolExecutor, Future

import time
def f(i):
    time.sleep(0.4)
    print('t-%s' % (i))
    time.sleep(0.4)

if __name__ == '__main__':
    a = time.time()
    
    # 注：线程池的上下文管理器退出时的 pool.shutdown() 默认是阻塞的
    with ThreadPoolExecutor(max_workers=10) as pool:
        for i in range(100):
            pool.submit(fn=f, i=i)
	# 等价于：
	# pool = ThreadPoolExecutor(max_workers=10)
    # for i in range(100):
    #     pool.submit(fn=f, i=i)
    # pool.shutdown(wait=True)
    
    print(time.time() - a)

```


获取任务的异常：（注：一般线程池的任务是不报错的）
```python
from concurrent.futures import ThreadPoolExecutor, Future

import time
def f(i):
    time.sleep(0.4)
    print('t-%s' % (i))
    time.sleep(0.4)

if __name__ == '__main__':
    a = time.time()
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [] # type: list[Future]
        for i in range(100):
            futures.append(pool.submit(fn=f, i=i))
        
        # 获取异常 (同步)
        for fut in futures:
            res = fut.exception()
            if res != None:
                raise res
    print(time.time() - a)
```




#### asyncio

概念
- 事件循环 `Event Loop`
- 协程 `Coroutine`
- 任务 `Task`
- 未来 `Future`

coroutine 是 async 封装的函数调用后的返回值（注：包含了使用的参数）
- `asyncio.run(<coroutine>)`：开始异步的主协程（注：会等待）
- `await <coroutine>`：在父协程下， 等待子协程 `<coroutine>`
- `asyncio.create_task(<coroutine>)`：将 `<coroutine>` 封装成 task 并开始执行任务


并发：
```python
import asyncio, time, random

async def f():
    while True:
        await asyncio.sleep(random.random()*2)
        print(asyncio.current_task().name, time.time())

async def main():
    t1 = asyncio.create_task(f())
    t2 = asyncio.create_task(f())
    t1.name = 't1'
    t2.name = 't2'
    await t1
    await t2

if __name__ == '__main__':
    asyncio.run(main())
```

---
可超时任务

```python
import asyncio
def timout_task(target, args=[], kwargs={}, timeout=1):
    async def run_function_async(func, *args, **kwargs):
        return await asyncio.get_running_loop().run_in_executor(None, func, *args, **kwargs)
    async def main():
        try:
            # 创建一个异步任务
            task = asyncio.create_task(run_function_async(target, *args, **kwargs))
            # 等待任务完成，超时则取消
            await asyncio.wait_for(task, timeout=timeout)
            return task.result()  # 返回任务结果
        except asyncio.TimeoutError:
            print("Task timed out")
            return None
    return asyncio.run(main())
```

#### typing


#### importlib

importlib 可以使用注入字符串的方式动态导入模块
注：可以解决环形模块依赖

```python
from importlib import import_module
import os
print(import_module('os.path')==os.path)
```

#### functools

偏函数
```python
from functools import partial
def f(a=0, b=1, c=2):
    print(a,b,c)
g = partial(f, a=10, b=20, c=30)
```

#### abc

abc 用于定义抽象基类

例子：
```python
from abc import ABC, abstractmethod

class hardware(ABC):
    # 1. 抽象方法
    @abstractmethod
    def connect(self):
        pass
    @abstractmethod
    def disConnect(self):
        pass
    
    # 2. 抽象属性
    @property
    @abstractmethod
    def isConnect(self):
        pass
    
    # 3. 抽象类方法
    @classmethod
    @abstractmethod
    def connectionNum(self):
        pass
    
    # 4. 抽象静态方法
    @staticmethod
    @abstractmethod
    def getHardwareName():
        pass
    
    
class robot(hardware):
    def connect(self):
        print('robot connect...')
    def disConnect(self):
        print('robot dis connect...')
    
    @property
    def isConnect(self):
        return True
    
    @classmethod
    def connectionNum(self):
        return 114514

    @staticmethod
    def getHardwareName():
        return 'robot'
    
rb = robot()
rb.connect()
rb.disConnect()
print(rb.isConnect)
print(rb.connectionNum())
print(rb.getHardwareName())
```


#### atexit

注册退出程序时的回调
```python
import atexit
atexit.register(lambda: print('exit program...'))
```

#### re

```
re.findall(r'asd', 'asauhfasduiassd')
re.match(r'.*asd.*', 'asauhfasduiassd')
```


#### glob


```python
import glob

# 获取指定模式匹配的文件路径
print(glob.glob('.vscode/*'))
```


#### os

```python
import os
# 获取当前工作目录
os.getcwd()
# 将路径的不同部分连接起来
os.path.join(path, *paths)

# 判断 文件/目录 是否存在
os.path.exists(path)
# 判断是否为目录，并且该目录存在
os.path.isdir(dirPath)
# 判断是否为文件，并且该文件存在
os.path.isfile(filePath)
```


#### sys

```python
import sys
# 命令行参数
sys.argv
# 模块搜索路径
sys.path
```


#### traceback
与异常 Exception 相关的库，用于获取错误信息

---
打印错误

注：不会打印 `try...catch` 代码块上层的调用关系

对比：
```python
try:
	1/0
except:
	# (1) 对日志友好
	print(traceback.format_exc())
	# (2) 对日志不友好
	traceback.print_exc()

"""
Traceback (most recent call last):
  File "e:\work\2024-10\20241104-09\sd3.py", line 3, in <module>
    1/0
ZeroDivisionError: division by zero
"""
```

---


```python
def inner_function():
    1/0
    
def middle_function():
    inner_function()

def outer_function():
    try:
        middle_function()
    except Exception as e:
		# (0) 最灵活
		import sys
        cls, e, tb = sys.exc_info()
        t = ''
        for v in traceback.extract_tb(tb):
            t += f' ({v.filename}:{v.lineno}) - {v.name}() -> {v.line}' + '\n'
        print(t)

        # (1) 较灵活
        t = ''
        for v in traceback.extract_tb(e.__traceback__):
            t += f' ({v.filename}:{v.lineno}) - {v.name}() -> {v.line}' + '\n'
        print(t)

        # (2) 不够灵活
        t = ''
        for v in traceback.format_tb(e.__traceback__):
            t += v
        print(t)


"""
 (e:\codespaces\myprojects\fm\Feature\logging\sd2.py:11) - outer_function() -> middle_function()
 (e:\codespaces\myprojects\fm\Feature\logging\sd2.py:7) - middle_function() -> inner_function()
 (e:\codespaces\myprojects\fm\Feature\logging\sd2.py:4) - inner_function() -> 1/0

 (e:\codespaces\myprojects\fm\Feature\logging\sd2.py:24) - outer_function() -> middle_function()
 (e:\codespaces\myprojects\fm\Feature\logging\sd2.py:20) - middle_function() -> inner_function()
 (e:\codespaces\myprojects\fm\Feature\logging\sd2.py:17) - inner_function() -> 1/0

  File "e:\codespaces\myprojects\fm\Feature\logging\sd2.py", line 24, in outer_function
    middle_function()
  File "e:\codespaces\myprojects\fm\Feature\logging\sd2.py", line 20, in middle_function
    inner_function()
  File "e:\codespaces\myprojects\fm\Feature\logging\sd2.py", line 17, in inner_function
    1/0
"""
```

改进：
```python


```


#### datetime

---
datetime
```python
from datetime import datetime
dt = datetime(2025, 1, 12, hour=11, minute=45, second=14)
print(dt)
```

---
time
```python
from datetime import time
t = time(11, 45, 14)
print(t)
```

---
timedelta
```python
from datetime import timedelta
td = timedelta(seconds=114514)
print(td)
```

#### time

```python
import time
time.time()
```

#### heapq

```python
import numpy as np

# ==========================================
# (1) 对于数字序列
arr = list(range(10))
np.random.shuffle(arr)
print(arr)

import heapq
# 1. 将数组转换为小根堆 (注：堆是完全二叉树)
heapq.heapify(arr)
print(arr)

# 2. push 元素到堆
heapq.heappush(arr, 10)
heapq.heappush(arr, 11)

# 3. 弹出根元素
print(heapq.heappop(arr), arr)

# 4. 弹出根元素, 并 push 新的元素
print(heapq.heapreplace(arr, 113), arr)

# 5. 获取 数组 的前 n 大/小 的元素
print(heapq.nlargest(4, arr))
print(heapq.nsmallest(4, arr))

# ==========================================
# (2) 对象序列
class Num:
    def __init__(self, num):
        self.num = num
    
    # 定义 序
    # 注: 也可以定义 __gt__()
    def __lt__(self, r: 'Num'):
        return self.num < r.num
    
    def __repr__(self):
        return 'Num(%s)' % (self.num)
arr = [Num(v) for v in range(10)]
np.random.shuffle(arr)
print(arr)
heapq.heapify(arr)
print(arr)
print(heapq.nsmallest(3, arr))


# ==========================================
# (3) 广义序列
arr = [{'a': {'b': {'c': v}}} for v in range(10)], key=lambda x: x['a']['b']['c']
print(heapq.nsmallest(3, arr))
```
注：堆的元素不一定是 数字，实际上堆的元素可以是 ==具有序的元素==


#### itertools

迭代工具（适用于预处理即将迭代的数据，而不消耗大量空间）


```python
import itertools

# ==========================================
# (1) 组合迭代
# A_3^3
print(list(itertools.permutations(range(3))))
# A_3^2
print(list(itertools.permutations(range(3), 2)))
# C_3^2
print(list(itertools.combinations(range(3), 2)))
# 笛卡尔积 [0,1,2] x ['a','b','c']
print(list(itertools.product(range(3), ['a', 'b', 'c'])))

# ==========================================
# 
# 生成无限循环序列 [0,1,2,3,0,1,2,3,0,1,...]
itertools.cycle(range(4))

# 前缀和
print(list(itertools.accumulate([1,1,4,5,1,4])))

# (广义上)合并多个可迭代对象
# 注：但是没有物理意义上合并, 只是方便迭代
print(list(itertools.chain(range(4), range(6, 10))))

arr = range(6)
# 过滤可迭代对象
print(list(itertools.compress(arr, (v%2==0 for v in arr))))
print(list(itertools.filterfalse(lambda x: x%2!=0, arr)))
```

#### collections

包含大量容器：Counter，defaultdict，OrderedDict，namedtuple，deque，UserDict，UserList，UserString

#### bisect

```python
import bisect

# 在有序数组 sorted_arr 中二分查找 v 的下标 (左偏 或 右偏)
bisect.bisect_left(sorted_arr, v)
bisect.bisect_right(sorted_arr, v)

# 在有序数组 sorted_arr 中插入 v
bisect.insort_left(sorted_arr, v)
bisect.insort_right(sorted_arr, v)
```


#### zipfile


例子：压缩目录为 .zip 文件
```python
from zipfile import ZipFile
import os
os.chdir(os.path.dirname(__file__))
            
def crack(src_dir, dist_file = None):
    if dist_file == None:
        name = os.path.split(src_dir)[-1]
        dist_file = name + '.zip'
		dist_file = os.path.join(os.path.dirname(src_dir), dist_file)
    with ZipFile(dist_file, 'w') as zipf:
        for dir, subdirs, subfiles in os.walk(src_dir):
            for f in subfiles:
                f_abs = os.path.join(dir, f)
                zipf.write(f_abs, os.path.relpath(f_abs, src_dir))

inp = input('src dir: ')
if os.path.exists(inp) and os.path.isdir(inp):
    crack(inp)
    print('success')
else:
    print('error')

```


#### pickle
对象序列化处理

#### sqlite3
python 内置数据库

```python
import sqlite3

handle = sqlite3.connect('data.db')

cursor = handle.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    age INTEGER NOT NULL
)""")


handle.commit()
handle.close()

```

#### csv

---
写入数据

```python
import csv

with open('sd.csv', 'w', newline='') as f:
    f = csv.writer(f)
    # f = csv.writer(f, delimiter='\t')
    # f = csv.writer(f, delimiter='|')
    f.writerow([1,2,3])
    f.writerow(['',5,6])
    f.writerows([
        [7,8,9],
        [10,11,12],
    ])
```


---
读取数据

```python
with open('sd.csv', 'r') as f:
    f = csv.reader(f, delimiter='\t')
    # f = csv.reader(f, delimiter='|')
    arr = [[w for w in v] for v in f]
    print(arr)
```


#### tempfile

```python
import tempfile
# 创建临时目录, 如: C:\Users\Administrator\AppData\Local\Temp\tmpsfq67zl1
tempdir = tempfile.mkdtemp()
# 生成未创建的临时文件, 如: C:\Users\Administrator\AppData\Local\Temp\tmpdf3nq2l5
tempfile = tempfile.mktemp()
# 生成已创建的临时文件, 如: C:\Users\Administrator\AppData\Local\Temp\tmpdf3nq2l5
_, tempfile = tempfile.mkstemp()
```

#### requests

```python
import requests

# 普通 GET 请求
res = requests.get(r'https://www.google.com/')
# 使用代理 (如: clash)
res = requests.get(r'https://www.google.com/', proxies={
	"http": "http://127.0.0.1:7897",
	"https": "http://127.0.0.1:7897",
})
print(res)
```

