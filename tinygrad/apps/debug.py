# This file holds function that when called will append a section of the tensor to the log file for reference
from tinygrad.tensor import Tensor


def clear_log_file():
    with open("tensor_logs.txt", "w") as f:
        f.write("Tensor Logs\n\n")

clear_log_file()

def log_tensor(tensor: Tensor, name: str):
  with open("tensor_logs.txt", "a") as f:
    f.write(f"Tensor: {name}\n")
    f.write(str(tensor.numpy()))
    f.write("\n\n")