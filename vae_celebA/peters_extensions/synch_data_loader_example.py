import time
from multiprocessing import Queue, Process

class PoisonPill:
    pass

def _async_queue_manager(gen_func, queue: Queue):
    for item in gen_func():
        queue.put(item)
    queue.put(PoisonPill)

def iter_asynchronously(gen_func):
    """ Given a generator function, make it asynchonous.  """
    q = Queue()
    p = Process(target=_async_queue_manager, args=(gen_func, q))
    p.start()
    while True:
        item = q.get()
        if item is PoisonPill:
            break
        else:
            yield item

def data_loader():
    for i in range(4):
        time.sleep(1)  # Simulated loading time
        yield i

def main():
    start = time.time()
    for data in iter_asynchronously(data_loader):
        time.sleep(1)  # Simulated processing time
        processed_data = -data*2
        print(f'At t={time.time()-start:.3g}, processed data {data} into {processed_data}')

if __name__ == '__main__':
    main()
