import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 1. BaseGraph 파일에서 PCM 읽기 (예시: BG2, Z=3)
# code_PCM: BaseGraph 파일(예: ./BaseGraph/BaseGraph2_Set1.txt)
code_PCM = np.loadtxt("./BaseGraph/BaseGraph2_Set0.txt", int, delimiter='\t')
Z = 3  # Lifting size

# PCM to H: BaseGraph 확장 (Z-folding)
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

H = expand_pcm(code_PCM, Z)
print(f"H shape: {H.shape}")
M, N = H.shape
K = N - M
print(f"H shape: {H.shape}, Code rate: {K/N:.3f}")

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

def ldpc_encode(info_bits, H):
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

# # 2. 간이 LDPC 인코딩 (systematic, 실제 H로 더 정교하게 가능)
# def ldpc_encode(info_bits, H):
#     N = H.shape[1]
#     K = info_bits.size
#     codeword = np.zeros(N, dtype=int)
#     codeword[:K] = info_bits
#     # 패리티 계산 (최소제곱, 실제는 더 정교함)
#     H_p = H[:, K:]
#     H_sys = H[:, :K]
#     try:
#         parity = np.linalg.lstsq(H_p, np.mod(-H_sys @ info_bits, 2), rcond=None)[0]
#         parity = np.round(parity) % 2
#         codeword[K:] = parity.astype(int)
#     except Exception:
#         codeword[K:] = 0  # fallback
#     return codeword

# 3. BPSK 변조, AWGN 채널, LLR 계산
def bpsk_mod(bits):
    return 1 - 2*bits

def awgn_channel(x, snr_db):
    snr_linear = 10**(snr_db/10)
    sigma = np.sqrt(1/(2*snr_linear))
    noise = sigma * np.random.randn(*x.shape)
    return x + noise

def llr_awgn(rx, snr_db):
    snr_linear = 10**(snr_db/10)
    sigma2 = 1/(2*snr_linear)
    return 2 * rx / sigma2

def check_syndrome(codeword, H):
    return np.mod(H @ codeword, 2)

# 4. 디코더 구현들
def sign(x):
    return np.where(x < 0, -1, 1)

def sp_decode(llr, H, num_iter=25):
    M, N = H.shape
    Lq = np.tile(llr, (M,1))
    Lr = np.zeros((M,N))
    for it in range(num_iter):
        for m in range(M):
            idx = np.where(H[m]==1)[0]
            for n in idx:
                others = np.delete(idx, np.where(idx==n))
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

def ms_decode(llr, H, num_iter=25):
    # Min-Sum: SP와 거의 동일, soft 연산 아님
    M, N = H.shape
    Lq = np.tile(llr, (M,1))
    Lr = np.zeros((M,N))
    for it in range(num_iter):
        for m in range(M):
            idx = np.where(H[m]==1)[0]
            for n in idx:
                others = np.delete(idx, np.where(idx==n))
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

def nms_decode(llr, H, alpha=0.8, num_iter=25):
    M, N = H.shape
    Lq = np.tile(llr, (M,1))
    Lr = np.zeros((M,N))
    for it in range(num_iter):
        for m in range(M):
            idx = np.where(H[m]==1)[0]
            for n in idx:
                others = np.delete(idx, np.where(idx==n))
                prod_sign = np.prod(sign(Lq[m, others]))
                min_abs = np.min(np.abs(Lq[m, others]))
                Lr[m, n] = alpha * prod_sign * min_abs
        for n in range(N):
            idx = np.where(H[:, n]==1)[0]
            Lq[idx, n] = llr[n] + np.sum(Lr[idx, n], axis=0) - Lr[idx, n]
        llr_total = llr + np.sum(Lr, axis=0)
        hard_dec = (llr_total < 0).astype(int)
        if np.all(check_syndrome(hard_dec, H)==0):
            break
    return hard_dec

def oms_decode(llr, H, beta=0.15, num_iter=25):
    M, N = H.shape
    Lq = np.tile(llr, (M,1))
    Lr = np.zeros((M,N))
    for it in range(num_iter):
        for m in range(M):
            idx = np.where(H[m]==1)[0]
            for n in idx:
                others = np.delete(idx, np.where(idx==n))
                prod_sign = np.prod(sign(Lq[m, others]))
                min_abs = np.min(np.abs(Lq[m, others]))
                Lr[m, n] = prod_sign * max(0, min_abs - beta)
        for n in range(N):
            idx = np.where(H[:, n]==1)[0]
            Lq[idx, n] = llr[n] + np.sum(Lr[idx, n], axis=0) - Lr[idx, n]
        llr_total = llr + np.sum(Lr, axis=0)
        hard_dec = (llr_total < 0).astype(int)
        if np.all(check_syndrome(hard_dec, H)==0):
            break
    return hard_dec

# 5. 시뮬레이션
snr_db_list = np.arange(0, 5.5, 0.5)
num_blocks = 500  # 1000 이상 추천(연산에 따라 조절)
num_iter = 25
decoders = {
    'SP': sp_decode,
    'MS': ms_decode,
    'NMS': lambda llr, H, num_iter: nms_decode(llr, H, alpha=0.8, num_iter=num_iter),
    'OMS': lambda llr, H, num_iter: oms_decode(llr, H, beta=0.15, num_iter=num_iter)
}
results = {d: {'bler':[], 'ber':[]} for d in decoders}

for snr_db in tqdm(snr_db_list, desc="SNR sweep"):
    for dec_name, decoder in decoders.items():
        bler, ber, total_bits, total_blocks = 0, 0, 0, 0
        for _ in range(num_blocks):
            info_bits = np.zeros(K, dtype=int)  # all-zero codeword
            codeword = ldpc_encode(info_bits, H)
            print("Syndrome sum:", np.mod(H @ codeword, 2).sum())
            print("Codeword:", codeword)
            tx = bpsk_mod(codeword)
            rx = awgn_channel(tx, snr_db)
            llr = llr_awgn(rx, snr_db)
            dec = decoder(llr, H, num_iter=num_iter)
            block_err = not np.all(dec[:K]==info_bits)
            bit_err = np.sum(dec[:K]!=info_bits)
            bler += block_err
            ber += bit_err
            total_bits += K
            total_blocks += 1
        val_bler = max(bler/total_blocks, 1e-6)
        val_ber = ber/total_bits
        print(f"SNR={snr_db:.2f} dB, Decoder={dec_name}, BLER={val_bler:.4f}, BER={val_ber:.4f}")
        results[dec_name]['bler'].append(val_bler)
        results[dec_name]['ber'].append(val_ber)

# 6. 결과 시각화
plt.figure(figsize=(8,5))
for dec_name in decoders:
    plt.semilogy(snr_db_list, results[dec_name]['bler'], '-o', label=f'BLER {dec_name}')
plt.xlabel("SNR (dB)")
plt.ylabel("BLER (Block Error Rate)")
plt.title(f"LDPC Decoder BLER comparison (N={N}, K={K}, Z={Z}, I={num_iter})")
plt.grid(True, which='both')
plt.legend()
plt.show()
