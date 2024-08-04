import torch
import torch.nn as nn
import todos
import pdb

import ggml, ggml.utils


def test_reshape(x):
    B, C, H, W = x.size()
    x = x.cpu().to(torch.float).numpy()

    # 1) Allocate a new context with 256 MB of memory
    params = ggml.ggml_init_params(mem_size=256 * 1024 * 1024, mem_buffer=None)
    ctx = ggml.ggml_init(params)

    input = ggml.utils.from_numpy(x, ctx)
    print("ggml input shape: ", ggml.utils.get_shape(input))

    # 2) Build graph and compute
    f = ggml.ggml_reshape_4d(ctx, input, W//16, H//16, 256*C, B)
    gf = ggml.ggml_new_graph(ctx)
    ggml.ggml_build_forward_expand(gf, f)

    # 3) Compute the graph
    ggml.ggml_graph_compute_with_ctx(ctx, gf, 1)
    # --------------------------------------------------------------------------------------

    # Get the output value
    print("ggml output shape: ", ggml.utils.get_shape(f))
    output = ggml.utils.to_numpy(f)
    output = torch.from_numpy(output).clone()

    # Free the context
    ggml.ggml_free(ctx)

    return output


def torch_2d_pixel_unshuffle(x, r = 16):
	# x.size() -- 256, 256
	H, W = x.size()
	x = x.reshape(H//r, r, W//r, r)
	x = x.permute(1, 3, 0, 2).continues()
	x = x.reshape(r*r, H//r, W//r)
	return x

def ggml_2d_pixel_unshuffle(x, r=16):
    H, W = x.size()
    x = x.cpu().to(torch.float).numpy()

    # 1) Allocate a new context of memory
    S = 3 * H * W * 4 # here 4 === sizeof(float32)
    params = ggml.ggml_init_params(mem_size=S, mem_buffer=None)
    ctx = ggml.ggml_init(params)

    input = ggml.utils.from_numpy(x, ctx)
    print("ggml input shape: ", ggml.utils.get_shape(input))

    # 2) Build graph
    x1 = ggml.ggml_reshape_4d(ctx, input, r, W//r, r, H//r)
    x2 = ggml.ggml_permute(ctx, x1, 2, 0, 3, 1)
    x3 = ggml.ggml_cont(ctx, x2) # contiguous x2 !!!
    x4 = ggml.ggml_reshape_3d(ctx, x3, W//r, H//r, r*r)
    f = ggml.ggml_reshape_4d(ctx, x4, W//r, H//r, r*r, 1) # [32, 32, 64, 1]

    gf = ggml.ggml_new_graph(ctx)
    ggml.ggml_build_forward_expand(gf, f)

    # 3) Compute the graph
    ggml.ggml_graph_compute_with_ctx(ctx, gf, 1)

    # Get the output
    print("ggml output shape: ", ggml.utils.get_shape(f))
    output = ggml.utils.to_numpy(f)
    output = torch.from_numpy(output).clone()

    # Free the context
    ggml.ggml_free(ctx)

    return output # output.size() -- [1, 64, 32, 32]

def test_pixel_unshuffle():
	x = torch.randn(1, 3, 256, 256)
	model = nn.PixelUnshuffle(8)
	with torch.no_grad():
		s = model(x)

	y1 = ggml_2d_pixel_unshuffle(x[0][0][:, :], r=8)
	y2 = ggml_2d_pixel_unshuffle(x[0][1][:, :], r=8)
	y3 = ggml_2d_pixel_unshuffle(x[0][2][:, :], r=8)
	y = torch.cat((y1, y2, y3), dim=1) # ggml_concat(ctx, a, b, dim)

	assert (s - y).abs().max() < 0.0001, "s == y"
	print("Test PixelUnshuffle OK !")

# 1)
# x = torch.randn(1, 3, 256, 256)
# y = test_reshape(x)
# print("Abs.max():", (y - x.reshape(1, 768, 16, 16)).abs().max())

# 2)
test_pixel_unshuffle()


