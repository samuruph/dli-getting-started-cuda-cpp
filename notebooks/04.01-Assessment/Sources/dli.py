import subprocess
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import numpy as np
import glob
import os


def compile_and_run(filenames, runtime_args):
    for file in glob.glob('/tmp/heat_*'):
        os.remove(file)

    if os.path.exists('/tmp/a.out'):
        os.remove('/tmp/a.out')

    # Links with binary runner
    command = ['nvcc', '-arch=native', '-std=c++17', '-O3', '--extended-lambda', '-o',
               '/tmp/a.out'] + filenames
    result = subprocess.run(command)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        return result.returncode

    command = ['/tmp/a.out']
    if runtime_args is not None:
        command += runtime_args

    return subprocess.run(command).returncode


def run(filenames, runtime_args=None):
    result = compile_and_run(filenames, runtime_args)
    if result != 0:
        print('Failed')
        return None

    img = None
    fig = plt.figure(figsize=(8, 8))
    fig.tight_layout()

    def drawframe(i):
        with open(f"/tmp/ez_{i}.bin", 'rb') as f:
            height, width = np.fromfile(f, dtype=np.int32, count=2)
            data = np.fromfile(f, dtype=np.float32, count=height * width)
            data = data.reshape((height, width))

        nonlocal img
        if img is None:
            img = plt.imshow(data, cmap='seismic', interpolation='none')
        else:
            img.set_data(data)
        return img,

    ani = animation.FuncAnimation(
        fig, drawframe, frames=100, interval=20, blit=True)

    plt.close(fig)  # Suppress the figure display
    return HTML(ani.to_html5_video())


def run_step_1():
    return run(["Sources/maxwell.cu", "/usr/local/bin/runner_1.a"])


def run_step_2():
    return run(["Sources/maxwell.cu", "/usr/local/bin/runner_2.a"])


def run_step_3():
    return run(["Sources/maxwell.cu", "Sources/coarse.cu", "/usr/local/bin/runner_3.a"])


def passes_step_1_with_n_of(n):
    print('Running Maxwell simulator with {}^2 cells'.format(n))
    print('--------------------------------------------\n')

    if compile_and_run(["Sources/maxwell.cu", "/usr/local/bin/runner_1.a"], [f"-N={n}", "-v=1"]) != 0:
        return False

    return True


def passes_step_2_with_n_of(n):
    print('Running Maxwell simulator with {}^2 cells'.format(n))
    print('--------------------------------------------\n')

    if compile_and_run(["Sources/maxwell.cu", "/usr/local/bin/runner_2.a"], [f"-N={n}", "-v=1"]) != 0:
        return False

    return True


def passes_step_3_with_n_of(n):
    print('Running Maxwell simulator with {}^2 cells'.format(n))
    print('--------------------------------------------\n')

    if compile_and_run(["Sources/maxwell.cu", "Sources/coarse.cu", "/usr/local/bin/runner_3.a"], [f"-N={n}", "-v=1"]) != 0:
        return False

    return True
