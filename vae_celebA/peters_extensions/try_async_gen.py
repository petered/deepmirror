# import asyncio
# import json
#
# import itertools
#
#
# async def subscribe(name):
#     for i in range(itertools.count(100)):
#         # _, text = await db.poll()
#         if b"NONE" == text:
#             await asyncio.sleep(0.1)
#         else:
#             yield i
#
#
# class TickBatcher(object):
#     def __init__(self, db_name):
#         self.one_batch = []
#         self.db_name = db_name
#
#     async def sub(self):
#         async for item in subscribe(self.db_name):
#             self.one_batch.append(item)
#
#
# # def timer(secs=1):
# #     """async timer decorator"""
# #     def _timer(f):
# #         async def wrapper(*args, **kwargs):
# #             while 1:
# #                 await asyncio.sleep(secs)
# #                 await f()
# #         return wrapper
# #     return _timer
#
#
# class TickBatcher:
#
#     def __init__(self, one_batch=10):
#         self.one_batch = one_batch
#
#     async def run(self):
#
#         # do work here
#         print(len(self.one_batch))
#         self.one_batch = []
#
#
# if __name__ == '__main__':
#     loop = asyncio.get_event_loop()
#
#     proc = TickBatcher()
#     loop.create_task(proc.sub())
#     loop.create_task(proc.run())
#
#     loop.run_forever()
#     loop.close()
import itertools
import asyncio
import time
from asyncio import Queue

async def dataloader(queue: Queue):
    for i in itertools.count(0):
        queue.put(i)
        time.sleep(1)
        # yield i

#
def consumer():

    q = Queue()

    loop = asyncio.get_event_loop()
    # task = asyncio.get_event_loop().create_task(dataloader(q))
    task = asyncio.Task(dataloader(q))
    asyncio.run_coroutine_threadsafe(task, loop)

    start_time = time.time()

    for i in q.get():
        print(f'Count at t={time.time()-start_time:.5g}: {i}')
        time.sleep(1.)
#
#
#
if __name__ == '__main__':

    consumer()
    # asyncio.

