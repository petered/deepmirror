import itertools
import time
import asyncio

async def write_nums(queue):

    for i in range(itertools.count(0)):
        time.sleep(1)
        queue.put(i)



def main():

    queue = asyncio.Queue(maxsize=3)

    task = asyncio.create_task(write_nums(f'worker-{i}', queue))

    while True:
        print(queue.get())


if __name__ == '__main__':
    main()
