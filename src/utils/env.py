def _prepare_environment():
    torch.cuda.empty_cache()
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    print("Total memory:", info.total)
    print("Free memory:", info.free)
    print("Used memory:", info.used)