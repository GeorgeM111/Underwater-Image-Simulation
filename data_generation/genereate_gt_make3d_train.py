import sys
import time
import joblib
from joblib import Parallel, delayed

from data_2 import generate_and_save_ricardo_image_make_3D


START_IDX = 0
END_IDX = 400
CHUNK_SIZE = 100


def main():
    chunks = [
        (st, min(st + CHUNK_SIZE, END_IDX))
        for st in range(START_IDX, END_IDX, CHUNK_SIZE)
    ]

    print("Python Version :", sys.version)
    print("Joblib Version :", joblib.__version__)
    n_cpu = min(joblib.cpu_count(), 4)
    print("The number of CPU is :", n_cpu)
    print(f"Generating indices [{START_IDX}, {END_IDX}) in {len(chunks)} chunk(s) of up to {CHUNK_SIZE}")

    t0 = time.time()
    with Parallel(n_jobs=n_cpu) as parallel:
        parallel(delayed(generate_and_save_ricardo_image_make_3D)(st, en) for st, en in chunks)
    print(f"Total computation time : {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()