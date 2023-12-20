import math
import numpy as np
from numba import cuda, float64

THREADS=128

@cuda.jit()
def _vert_proj(input, output):
  x = cuda.threadIdx.x
  b = cuda.blockIdx.x
  i = cuda.grid(1)
  sv = cuda.shared.array((4, THREADS), dtype=float64)

  sv[0, x] = float64(0.)
  sv[1, x] = float64(0.)
  sv[2, x] = float64(0.)
  sv[3, x] = float64(0.)
  # each thread calculates a vertical projection
  if i < input.shape[1]:
    temp = float64(0.0)
    for j in range(input.shape[0]):
      temp += input[j, i]
    temp_1 = temp*i
    temp_2 = temp_1*i
    sv[0, x] = temp
    sv[1, x] = temp_1
    sv[2, x] = temp_2
    sv[3, x] = temp_2*i

    cuda.syncthreads()

    l = THREADS // 2
    while (x < l):
      sv[0, x] = sv[0, x] + sv[0, x+l]
      sv[1, x] = sv[1, x] + sv[1, x+l]
      sv[2, x] = sv[2, x] + sv[2, x+l]
      sv[3, x] = sv[3, x] + sv[3, x+l]
      l = l // 2
      cuda.syncthreads()
    if x == 0:
      output[0, b] = sv[0, 0]
      output[1, b] = sv[1, 0]
      output[2, b] = sv[2, 0]
      output[3, b] = sv[3, 0]
  return


@cuda.jit()
def _horz_proj(input, output):
  x = cuda.threadIdx.x
  b = cuda.blockIdx.x
  i = cuda.grid(1)
  sh = cuda.shared.array((3, THREADS), dtype=float64)

  sh[0, x] = float64(0.)
  sh[1, x] = float64(0.)
  sh[2, x] = float64(0.)
  # each thread calculates a horizontal projection
  if i < input.shape[0]:
    temp = float64(0.)
    for j in range(input.shape[1]):
      temp += input[i, j]
    temp_1 = temp*i
    temp_2 = temp_1*i
    sh[0, x] = temp_1
    sh[1, x] = temp_2
    sh[2, x] = temp_2*i

    cuda.syncthreads()

    l = THREADS // 2
    while (x < l):
      sh[0, x] = sh[0, x] + sh[0, x+l]
      sh[1, x] = sh[1, x] + sh[1, x+l]
      sh[2, x] = sh[2, x] + sh[2, x+l]
      l = l // 2
      cuda.syncthreads()
    if x == 0:
      output[4, b] = sh[0, 0]
      output[5, b] = sh[1, 0]
      output[6, b] = sh[2, 0]
  return


@cuda.jit()
def _diag_proj(input, output):
  x = cuda.threadIdx.x
  blockdim = cuda.blockDim.x
  b = cuda.blockIdx.x
  k = cuda.grid(1)
  sd = cuda.shared.array((2, THREADS), dtype=float64)

  sd[0, x] = float64(0.)
  sd[1, x] = float64(0.)
  # want i+j=k
  # iterate over i
  # input[i, k-i]
  # account for when k > shape[1]-1

  if k < input.shape[0]+input.shape[1]:
    temp = float64(0.)
    start = 0
    end = k
    if k > input.shape[1]-1:
      start = k - (input.shape[1]-1)
    if k > input.shape[0] - 1:
      end = input.shape[0] - 1
    while start < end+1:
      temp += input[start, k-start]
      start+=1

    temp_2 = temp*(k**2)
    sd[0, x] = temp_2
    sd[1, x] = temp_2*k

    cuda.syncthreads()
    l = blockdim // 2
    while (x < l):
      sd[0, x] = sd[0, x] + sd[0, x+l]
      sd[1, x] = sd[1, x] + sd[1, x+l]
      l = l // 2
      cuda.syncthreads()
    if x == 0:
      output[7, b] = sd[0, 0]
      output[8, b] = sd[1, 0]
  return

@cuda.jit()
def _anti_proj(input, output):
  x = cuda.threadIdx.x
  blockdim = cuda.blockDim.x
  b = cuda.blockIdx.x
  k = cuda.grid(1)
  sad = cuda.shared.array((1, THREADS), dtype=float64)

  sad[0, x] = float64(0.)
  # want i-j+M-1 = k
  # where M in shape[1]
  # iterate over i
  # input[i, M-1+i-k]

  if k < input.shape[0] + input.shape[1]:
    temp = float64(0.)
    m = input.shape[1]
    start = 0
    end = k
    if k > input.shape[1]-1:
      start = k - (input.shape[1]-1)
    if k > input.shape[0] - 1:
      end = input.shape[0] - 1
    while start <= end:
      temp += input[start, m-1+start-k]
      start+=1

    sad[0, x] = temp*((k-m+1)**3)

    cuda.syncthreads()
    l = blockdim // 2
    while (x < l):
      sad[0, x] = sad[0, x] + sad[0, x+l]
      l = l // 2
      cuda.syncthreads()
    if x == 0:
      output[9, b] = sad[0, 0]
  return


def gpu_moments(input):
  # 10 projections:
  # vert, vert_1, vert_2, vert_3
  # hor_1, hor_2, hor_3
  # d_2, d_3
  # a_3

  diag_block_size = math.ceil((input.shape[0]+input.shape[1])/THREADS)

  proj_array = np.zeros(shape=(10, diag_block_size), dtype=np.float64)
  proj = cuda.to_device(proj_array)

  _vert_proj[diag_block_size//2, THREADS](input, proj)
  _horz_proj[diag_block_size//2, THREADS](input, proj)
  _diag_proj[diag_block_size, THREADS](input, proj)
  _anti_proj[diag_block_size, THREADS](input, proj)
  p = proj.copy_to_host()
  return p

def moments(input):
  proj = gpu_moments(input)
  m = {}
  m["m00"] = np.sum(proj[0])
  m["m10"] = np.sum(proj[1])
  m["m20"] = np.sum(proj[2])
  m["m30"] = np.sum(proj[3])
  m["m01"] = np.sum(proj[4])
  m["m02"] = np.sum(proj[5])
  m["m03"] = np.sum(proj[6])
  m["m11"] = (np.sum(proj[7]) - m["m20"] - m["m02"])/2.0
  d3 = np.sum(proj[8])/6.0
  a3 = np.sum(proj[9])/6.0
  m["m12"] = (d3 - a3) - m["m30"]/3.0
  m["m21"] = (d3 + a3) - m["m03"]/3.0
  return m