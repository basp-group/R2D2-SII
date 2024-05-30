from lib.operator import Operator

def gen_op_R2D2Net(im_size, data_dict, device):
    # generate PSF which is needed only for R2D2Net/ R3D3
    op_R2D2Net = Operator(im_size=im_size, device=device)
    op_R2D2Net.set_uv_imweight(data_dict['uv'], data_dict['nWimag'])
    return op_R2D2Net