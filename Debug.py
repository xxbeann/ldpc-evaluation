import numpy as np

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
            # cyclically shifted identity matrix
            I = np.eye(Z, dtype=int)
            shift = entry % Z
            block = np.roll(I, shift, axis=1)
            H[m*Z:(m+1)*Z, n*Z:(n+1)*Z] = block
    return H

H = expand_pcm(base_pcm, Z)
M, N = H.shape
K = N - M
print(f"Expanded H: {H.shape}, Code rate={K/N:.3f}")

# 2. 인코딩 함수
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

# 5. SNR sweep 예시
snr_db_list = [0.3, 0.5, 0.7]  # 짧은 테스트
for snr_db in snr_db_list:
    info_bits = np.zeros(K, dtype=int)  # all-zero word
    codeword = ldpc_encode_systematic(info_bits, H)
    print("Syndrome sum:", check_syndrome(codeword, H).sum())
    tx = bpsk_mod(codeword)
    rx = awgn_channel(tx, snr_db)
    llr = llr_awgn(rx, snr_db)
    dec = ms_decode(llr, H, num_iter=25)
    print(f"SNR={snr_db:.1f}dB, Block Error={(dec[:K]!=info_bits).any()}")

# 필요시 여러 블록 반복, BLER/BER 평가/그래프 추가