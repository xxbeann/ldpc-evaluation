import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

# 1. BaseGraph2_Set0 읽기
base_pcm = np.loadtxt("./BaseGraph/BaseGraph2_Set0.txt", dtype=int, delimiter='\t')
Z = 3  # 원하는 lifting size

def expand_pcm(base_pcm, Z):
    M_base, N_base = base_pcm.shape
    M, N = M_base * Z, N_base * Z
    H = np.zeros((M, N), dtype=int)
    for m in range(M_base):
        for n in range(N_base):
            entry = base_pcm[m, n]
            if entry == -1:
                continue  # zero block
            I = np.eye(Z, dtype=int)
            shift = entry % Z
            block = np.roll(I, shift, axis=1)
            H[m*Z:(m+1)*Z, n*Z:(n+1)*Z] = block
    return H

H = expand_pcm(base_pcm, Z)
M, N = H.shape
K = N - M
print(f"Expanded H: {H.shape}, Code rate={K/N:.3f}")

# 2. 인코딩 함수 (변경 없음)
def solve_gf2(A, b):
    A = A.copy()
    b = b.copy()
    n = A.shape[1]
    m = A.shape[0]
    x = np.zeros(n, dtype=int)
    row = 0
    for col in range(n):
        pivot = None
        for r in range(row, m):
            if A[r, col] == 1:
                pivot = r
                break
        if pivot is None:
            continue
        if pivot != row:
            A[[row, pivot]] = A[[pivot, row]]
            b[[row, pivot]] = b[[pivot, row]]
        for r in range(row+1, m):
            if A[r, col] == 1:
                A[r] ^= A[row]
                b[r] ^= b[row]
        row += 1
        if row == m:
            break
    for i in range(n-1, -1, -1):
        rows_with_1 = np.where(A[:, i]==1)[0]
        if len(rows_with_1)==0:
            x[i] = 0
        else:
            row = rows_with_1[0]
            s = 0
            for j in range(i+1, n):
                s ^= (A[row, j] & x[j])
            x[i] = (b[row] ^ s) % 2
    return x

def ldpc_encode_systematic(info_bits, H):
    N = H.shape[1]
    M = H.shape[0]
    K = N - M
    assert info_bits.size == K, "info_bits 크기 오류"
    H_sys = H[:, :K]
    H_p = H[:, K:]
    rhs = (-H_sys @ info_bits) % 2
    parity_bits = solve_gf2(H_p, rhs)
    codeword = np.concatenate([info_bits, parity_bits])
    return codeword

# 3. 부가 함수
def bpsk_mod(bits): return 1 - 2*bits
def awgn_channel(x, snr_db):
    snr_linear = 10**(snr_db/10)
    sigma = np.sqrt(1/(2*snr_linear))
    return x + sigma * np.random.randn(*x.shape)
def llr_awgn(rx, snr_db):
    snr_linear = 10**(snr_db/10)
    sigma2 = 1/(2*snr_linear)
    return 2 * rx / sigma2
def check_syndrome(codeword, H): return np.mod(H @ codeword, 2)

# 4. 디코더 예시 (Min-Sum)
def sign(x): return np.where(x < 0, -1, 1)
def ms_decode(llr, H, num_iter=500):
    M, N = H.shape
    Lq = np.tile(llr, (M,1))
    Lr = np.zeros((M,N))
    for it in range(num_iter):
        for m in range(M):
            idx = np.where(H[m]==1)[0]
            for n in idx:
                others = np.delete(idx, np.where(idx==n))
                if len(others) == 0:
                    Lr[m, n] = 0
                else:
                    prod_sign = np.prod(sign(Lq[m, others]))
                    min_abs = np.min(np.abs(Lq[m, others]))
                    Lr[m, n] = prod_sign * min_abs
        for n in range(N):
            idx = np.where(H[:, n]==1)[0]
            Lq[idx, n] = llr[n] + np.sum(Lr[idx, n], axis=0) - Lr[idx, n]
        llr_total = llr + np.sum(Lr, axis=0)
        hard_dec = (llr_total < 0).astype(int)
        if np.all(check_syndrome(hard_dec, H)==0):
            break
    return hard_dec

# 시뮬레이션 파라미터
snr_db_list = np.arange(-4, 0, 0.2)
num_blocks = 20000  # 실제 논문 수준 실험은 500~10000 정도 권장
num_iter = 25

# 병렬화 함수: 한 SNR에서의 블록 실험
def simulate_one_block(snr_db, K, H, num_iter):
    info_bits = np.zeros(K, dtype=int)  # all-zero word (성능 worst-case 아님. 랜덤도 가능)
    codeword = ldpc_encode_systematic(info_bits, H)
    tx = bpsk_mod(codeword)
    rx = awgn_channel(tx, snr_db)
    llr = llr_awgn(rx, snr_db)
    dec = ms_decode(llr, H, num_iter=num_iter)
    block_error = not np.all(dec[:K] == info_bits)
    bit_error = np.sum(dec[:K] != info_bits)
    return block_error, bit_error

# 전체 시뮬레이션 (SNR별로 병렬 처리)
ber_list = []
bler_list = []
n_jobs = multiprocessing.cpu_count()  # 사용 가능한 코어 수

for snr_db in tqdm(snr_db_list, desc="SNR sweep"):
    # block 단위 병렬화
    results = Parallel(n_jobs=n_jobs)(
        delayed(simulate_one_block)(snr_db, K, H, num_iter)
        for _ in range(num_blocks)
    )
    block_errors = sum(r[0] for r in results)
    bit_errors = sum(r[1] for r in results)
    bler = max(block_errors / num_blocks, 1e-7)
    ber = max(bit_errors / (num_blocks * K), 1e-9)
    bler_list.append(bler)
    ber_list.append(ber)
    print(f"SNR={snr_db:.2f}dB, BLER={bler:.4e}, BER={ber:.4e}")

# --- 결과 시각화 ---
plt.figure(figsize=(8,5))
plt.semilogy(snr_db_list, bler_list, '-o', label='BLER (Block Error Rate)')
plt.semilogy(snr_db_list, ber_list, '-s', label='BER (Bit Error Rate)')
plt.xlabel("SNR (dB)")
plt.ylabel("Error Rate")
plt.title(f"LDPC MS-Decoder\n(N={N}, K={K}, Z={Z}, Iter={num_iter})")
plt.grid(True, which='both')
plt.legend()
plt.tight_layout()
plt.show()