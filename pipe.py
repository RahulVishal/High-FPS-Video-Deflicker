import cv2
import numpy as np
import torch
import torch.nn as nn
from time import time
import threading, queue

# Dask imports
from dask.distributed import Client, wait

# Top-level parameters
INPUT_VIDEO_PATH = r"videos\Example Video With Flickering (360p).mp4"
OUTPUT_VIDEO_PATH = "deflicker_output_dask.mp4"
WINDOW_LENGTH = 16        # Number of frames per window
NUM_PROC_WORKERS = 4      # Number of workers for processing (threads for non-cuda, processes for dask-cuda)
MAX_ACTIVE_FUTURES = 50   # Maximum tasks held in RAM at once
USE_DASK_CUDA = True     # Set to True to use dask-cuda (requires dask_cuda)

# Identity model as before
class IdentityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Identity()
        print("[✓] Initialized identity model")
    def forward(self, x):
        return x

# The processing function now takes extra parameters so that each worker knows the image dimensions, window length,
# and gets a reference to the model and device.
def process_window(item, model, device, height, width, window_length):
    index, window = item
    tensor = torch.from_numpy(window).float() / 255.0
    tensor = tensor.unsqueeze(0).to(device, non_blocking=True)
    with torch.no_grad():
        output = model(tensor).cpu().numpy()
    processed = (output * 255).astype(np.uint8).reshape(window_length, height, width, 3)
    return (index, processed)

def process_video(input_path, output_path, window_length, num_proc_workers, max_active_futures, use_dask_cuda):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video specs: {width}x{height} @ {fps:.2f} FPS")
    
    # Queues for pipelining: one for reading windows, one for writing output windows.
    window_queue = queue.Queue(maxsize=100)
    output_queue = queue.Queue(maxsize=100)
    
    # Reader thread: groups frames into windows.
    def reader():
        window = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB.
            window.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if len(window) == window_length:
                window_queue.put((idx, np.array(window)))
                idx += 1
                window = []
        cap.release()
        # Push one sentinel per worker so the processing loop knows reading is done.
        for _ in range(num_proc_workers):
            window_queue.put(None)
    
    # Writer thread: writes processed windows in order.
    def writer():
        next_idx = 0
        buffer = {}
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        while True:
            item = output_queue.get()
            if item is None:
                break
            index, proc_window = item
            buffer[index] = proc_window
            # Write windows in order as soon as possible.
            while next_idx in buffer:
                for frame in buffer.pop(next_idx):
                    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                next_idx += 1
        # Flush any remaining frames.
        while next_idx in buffer:
            for frame in buffer.pop(next_idx):
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            next_idx += 1
        out.release()
        print("\nVideo saved successfully!")
    
    # Set device and initialize model.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = IdentityModel().to(device)
    model.eval()

    # Create the Dask client.
    if use_dask_cuda:
        from dask_cuda import LocalCUDACluster
        cluster = LocalCUDACluster(n_workers=num_proc_workers)
        client = Client(cluster)
    else:
        # Use threads so that the same model instance is shared (processes=False).
        client = Client(processes=False, threads_per_worker=1, n_workers=num_proc_workers)
    
    # For dask-cuda (process-based) scatter the model and device so that each worker gets its own copy.
    if use_dask_cuda:
        scattered_model = client.scatter(model, broadcast=True)
        scattered_device = client.scatter(device, broadcast=True)
    else:
        scattered_model = model
        scattered_device = device

    start_time = time()
    reader_thread = threading.Thread(target=reader)
    writer_thread = threading.Thread(target=writer)
    reader_thread.start()
    writer_thread.start()
    
    # Use Dask to submit tasks. We mimic the original “active futures” approach.
    active_futures = []
    finished_reading = False
    while not finished_reading or active_futures:
        # Fill the active_futures list until the limit is reached.
        while not finished_reading and len(active_futures) < max_active_futures:
            item = window_queue.get()
            if item is None:
                finished_reading = True
                break
            # Submit the processing task to Dask.
            future = client.submit(
                process_window, item, scattered_model, scattered_device,
                height, width, window_length
            )
            active_futures.append(future)
        if active_futures:
            # Wait until at least one future is complete.
            done, active_futures = wait(active_futures, return_when='FIRST_COMPLETED')
            for future in done:
                output_queue.put(future.result())
    
    # Signal writer thread that processing is complete.
    output_queue.put(None)
    reader_thread.join()
    writer_thread.join()
    client.close()
    print(f"\nTotal processing time: {time() - start_time:.2f} seconds")

if __name__ == '__main__':
    process_video(INPUT_VIDEO_PATH, OUTPUT_VIDEO_PATH, WINDOW_LENGTH,
                  NUM_PROC_WORKERS, MAX_ACTIVE_FUTURES, USE_DASK_CUDA)
    print("\nDownload processed video from:", OUTPUT_VIDEO_PATH)
