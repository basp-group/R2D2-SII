from data import transforms as T

import torch

def forward(args, i, net, res_n, output_n, data, mean, op_R2D2Net=None):
    if args.layers == 1:
        if i == 0:
            output = net(torch.cat((res_n, output_n), dim=1))
        else:
            output = net(torch.cat((output_n, res_n.to(args.device)), dim=1)) + output_n
    else:
        for j in range(args.layers):
            if j == 0:
                output = net[j](torch.cat((res_n.to(args.device), output_n), dim=1)) + output_n
            else:
                output = net[j](torch.cat((res_tmp, output), dim=1)) + output
                del res_tmp
            if j < (args.layers - 1):
                output *= (mean.to(args.device) + 1e-110)
                res_tmp = op_R2D2Net.gen_res(data['dirty'], output, PSF=data['PSF'])
                output, mean = T.normalize_instance(output, eps=1e-110)
                res_tmp = T.normalize(res_tmp, mean, eps=1e-110)
    output = torch.clip(output * (mean + 1e-110), min=0, max=None)
    return output