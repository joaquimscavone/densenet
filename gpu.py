from pynvml import *
from tensorflow.python.client import device_lib
import os
nvmlInit()
devices_id = [0]*nvmlDeviceGetCount()
for i in range(len(devices_id)):
    handle = nvmlDeviceGetHandleByIndex(i)
    meminfo = nvmlDeviceGetMemoryInfo(handle)
    print("%s: %0.1f MB free, %0.1f MB used, %0.1f MB total" % (nvmlDeviceGetName(handle),meminfo.free/1024.**2, meminfo.used/1024.**2, meminfo.total/1024.**2))
    print(device_lib.list_local_devices())
nvmlShutdown()