import numpy as np


def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
	# First figure out what the size of the output should be
	N, C, H, W = x_shape
	assert (H + 2 * padding - field_height) % stride == 0
	assert (W + 2 * padding - field_height) % stride == 0
	out_height = int((H + 2 * padding - field_height) // stride + 1)
	out_width = int((W + 2 * padding - field_width) // stride + 1)

	i0 = np.repeat(np.arange(field_height), field_width)
	i0 = np.tile(i0, C)
	i1 = stride * np.repeat(np.arange(out_height), out_width)
	j0 = np.tile(np.arange(field_width), field_height * C)
	j1 = stride * np.tile(np.arange(out_width), out_height)
	i = i0.reshape(-1, 1) + i1.reshape(1, -1)
	j = j0.reshape(-1, 1) + j1.reshape(1, -1)

	k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

	return (k.astype(int), i.astype(int), j.astype(int))


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
	""" An implementation of im2col based on some fancy indexing """
	# Zero-pad the input
	p = padding
	x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

	k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)

	cols = x_padded[:, k, i, j]
	C = x.shape[1]
	cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
	return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,
				   stride=1):
	""" An implementation of col2im based on fancy indexing and np.add.at """
	N, C, H, W = x_shape
	H_padded, W_padded = H + 2 * padding, W + 2 * padding
	x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
	k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
	cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
	cols_reshaped = cols_reshaped.transpose(2, 0, 1)
	np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
	if padding == 0:
		return x_padded
	return x_padded[:, :, padding:-padding, padding:-padding]


def conv_forward_fast(X, W, b, conv_param):
	X = np.transpose(X,(0,3,1,2))
	W = np.transpose(W,(0,3,1,2))
	#b2 = np.array([b]).T
	b2 = b[None].T
	padding = conv_param["pad"]
	stride = conv_param["stride"]
	
	n_filters, d_filter, h_filter, w_filter = W.shape
	n_x, d_x, h_x, w_x = X.shape
	h_out = (h_x - h_filter + 2 * padding) / stride + 1
	w_out = (w_x - w_filter + 2 * padding) / stride + 1

	if not h_out.is_integer() or not w_out.is_integer():
		raise Exception('Invalid output dimension!')

	h_out, w_out = int(h_out), int(w_out)

	X_col = im2col_indices(X, h_filter, w_filter, padding=padding, stride=stride)
	W_col = W.reshape(n_filters, -1)

	out = W_col @ X_col + b2
	out = out.reshape(n_filters, h_out, w_out, n_x)
	out = out.transpose(3, 0, 1, 2)

	X = np.transpose(X,(0,2,3,1))
	W = np.transpose(W,(0,2,3,1))
	out = np.transpose(out,(0,2,3,1))
	cache = (X, W, b, conv_param, X_col)
	return out, cache


def conv_backward_fast(dout, cache):
	X, W, b, conv_param, X_col = cache
	X = np.transpose(X,(0,3,1,2))
	W = np.transpose(W,(0,3,1,2))
	dout = np.transpose(dout,(0,3,1,2))
	n_filter, d_filter, h_filter, w_filter = W.shape

	padding = conv_param["pad"]
	stride = conv_param["stride"]

	db = np.sum(dout, axis=(0, 2, 3))
	db = db.reshape(n_filter, -1)

	dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(n_filter, -1)
	dW = dout_reshaped @ X_col.T
	dW = dW.reshape(W.shape)

	W_reshape = W.reshape(n_filter, -1)
	dX_col = W_reshape.T @ dout_reshaped
	dX = col2im_indices(dX_col, X.shape, h_filter, w_filter, padding=padding, stride=stride)

	dX = np.transpose(dX,(0,2,3,1))
	dW = np.transpose(dW,(0,2,3,1))
	db = db.T[0]
	return dX, dW, db


def pool_forward_fast(X, pool_param, mode='max'):
	def maxpool(X_col):
		max_idx = np.argmax(X_col, axis=0)
		out = X_col[max_idx, range(max_idx.size)]
		return out, max_idx
		
	def avgpool(X_col):
		out = np.mean(X_col, axis=0)
		cache = None
		return out, cache
		
	if mode == 'max':
		return _pool_forward_fast(X, maxpool, pool_param)
	elif mode == 'avg':
		return _pool_forward_fast(X, avgpool, pool_param)
	
	
def pool_backward_fast(dout, cache, mode = 'max'):
	def dmaxpool(dX_col, dout_col, pool_cache):
		dX_col[pool_cache, range(dout_col.size)] = dout_col
		return dX_col
		
	def davgpool(dX_col, dout_col, pool_cache):
		dX_col[:, range(dout_col.size)] = 1. / dX_col.shape[0] * dout_col
		return dX_col

	if mode == 'max':
		return _pool_backward_fast(dout, dmaxpool, cache)
	elif mode == 'avg':
		return _pool_backward_fast(dout, davgpool, cache)
	

	
def _pool_forward_fast(X, pool_fun, pool_param):

	X = np.transpose(X,(0,3,1,2))
	
	pool_height = pool_param['pool_height']
	pool_width = pool_param['pool_width']
	stride = pool_param['stride']	
	
	
	n, d, h, w = X.shape
	h_out = (h - pool_height) / stride + 1
	w_out = (w - pool_width) / stride + 1

	if not w_out.is_integer() or not h_out.is_integer():
		raise Exception('Invalid output dimension!')

	h_out, w_out = int(h_out), int(w_out)

	X_reshaped = X.reshape(n * d, 1, h, w)
	X_col = im2col_indices(X_reshaped, pool_height, pool_width, padding=0, stride=stride)

	out, pool_cache = pool_fun(X_col)

	out = out.reshape(h_out, w_out, n, d)
	#out = out.transpose(2, 3, 0, 1)

	X = np.transpose(X,(0,2,3,1))
	out = out.transpose(2,0,1,3)
	cache = (X, pool_param, X_col, pool_cache)

	return out, cache


def _pool_backward_fast(dout, dpool_fun, cache):
	X, pool_param, X_col, pool_cache = cache
	
	pool_height = pool_param['pool_height']
	pool_width = pool_param['pool_width']
	stride = pool_param['stride']
	
	X = np.transpose(X,(0,3,1,2))
	dout = np.transpose(dout,(0,3,1,2))
	
	n, d, w, h = X.shape

	dX_col = np.zeros_like(X_col)
	dout_col = dout.transpose(2, 3, 0, 1).ravel()

	dX = dpool_fun(dX_col, dout_col, pool_cache)

	dX = col2im_indices(dX_col, (n * d, 1, h, w), pool_height, pool_width, padding=0, stride=stride)
	dX = dX.reshape(X.shape)
	
	dX = np.transpose(dX,(0,2,3,1))
	
	return dX