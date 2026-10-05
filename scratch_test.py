import pywt
import numpy as np

def modwt_canonical(x, wavelet_name, levels):
    x = np.asarray(x, dtype=float)
    wavelet = pywt.Wavelet(wavelet_name)
    h = np.array(wavelet.dec_hi) / np.sqrt(2)
    g = np.array(wavelet.dec_lo) / np.sqrt(2)
    
    v = x.copy()
    w_list = []
    
    for j in range(levels):
        # Circular convolution for MODWT
        # Upsample filters by inserting 2^j - 1 zeros between elements
        step = 2**j
        N = len(x)
        
        v_next = np.zeros(N)
        w_next = np.zeros(N)
        
        # Filter length
        L = len(h)
        
        for t in range(N):
            # Applying filter circularly
            v_val = 0.0
            w_val = 0.0
            for l in range(L):
                idx = (t - l * step) % N
                v_val += g[l] * v[idx]
                w_val += h[l] * v[idx]
            v_next[t] = v_val
            w_next[t] = w_val
            
        w_list.append(w_next)
        v = v_next
        
    return w_list, v

def imodwt_canonical(w_list, v, wavelet_name):
    wavelet = pywt.Wavelet(wavelet_name)
    h = np.array(wavelet.rec_hi) / np.sqrt(2)
    g = np.array(wavelet.rec_lo) / np.sqrt(2)
    
    levels = len(w_list)
    N = len(v)
    
    for j in range(levels - 1, -1, -1):
        v_prev = np.zeros(N)
        w = w_list[j]
        step = 2**j
        L = len(h)
        
        for t in range(N):
            val = 0.0
            for l in range(L):
                idx = (t + l * step) % N
                val += g[l] * v[idx] + h[l] * w[idx]
            v_prev[t] = val
        v = v_prev
        
    return v

# Test canonical
np.random.seed(42)
x = np.random.randn(101)
w, v = modwt_canonical(x, 'db4', 4)
rec = imodwt_canonical(w, v, 'db4')
print('Canonical reconstruction error:', np.max(np.abs(x - rec)))

# Now to get MRA details: D_j is imodwt(zeros except w_j, zero_v)
D_list = []
for j in range(4):
    w_temp = [np.zeros(101) for _ in range(4)]
    w_temp[j] = w[j]
    D_j = imodwt_canonical(w_temp, np.zeros(101), 'db4')
    D_list.append(D_j)
S = imodwt_canonical([np.zeros(101) for _ in range(4)], v, 'db4')
rec_mra = sum(D_list) + S
print('Canonical MRA reconstruction error:', np.max(np.abs(x - rec_mra)))

x_shift = np.roll(x, 5)
w_shift, v_shift = modwt_canonical(x_shift, 'db4', 4)
D_shift_list = []
for j in range(4):
    w_temp = [np.zeros(101) for _ in range(4)]
    w_temp[j] = w_shift[j]
    D_j = imodwt_canonical(w_temp, np.zeros(101), 'db4')
    D_shift_list.append(D_j)
S_shift = imodwt_canonical([np.zeros(101) for _ in range(4)], v_shift, 'db4')

print('Canonical Shift invariance max diff D1:', np.max(np.abs(np.roll(D_list[0], 5) - D_shift_list[0])))

