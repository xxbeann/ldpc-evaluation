# import numpy
# numpy.show_config()  # BLAS/MKL 등 정보 출력

import numpy as np
import time

a = np.random.randn(8000, 8000)
b = np.random.randn(8000, 8000)

print("곱셈 시작, Activity Monitor에서 CPU 확인!")
start = time.time()
c = np.dot(a, b)
print("끝:", time.time()-start)

# import numpy as np
# import ctypes

# try:
#     lib = np.__config__.get_info('blas_opt_info').get('libraries', [''])[0]
#     if 'openblas' in lib:
#         # OpenBLAS는 일반적으로 libopenblas.dylib로 로딩
#         libopenblas = ctypes.cdll.LoadLibrary('libopenblas.dylib')
#         print("OpenBLAS 스레드 수:", libopenblas.openblas_get_num_threads())
# except Exception as e:
#     print("OpenBLAS 스레드 수 확인 실패:", e)