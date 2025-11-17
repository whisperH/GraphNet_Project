import argparse

from matplotlib import pyplot as plt
import torch

parser = argparse.ArgumentParser(description='Plot the W from a CRF model')
parser.add_argument('ckpt_path', default="", metavar='CKPT_PATH', type=str,
                    help='Path to the ckpt file of a CRF model')


def main():
    args = parser.parse_args()
    ckpt = torch.load(args.ckpt_path)
    
    print(ckpt['state_dict'].keys())
    print(ckpt['state_dict']['module.level_crf_model.W'].cpu().numpy(), ckpt['state_dict']['module.level_crf_model.W'].cpu().numpy().shape)
    W = ckpt['state_dict']['module.level_crf_model.W'].cpu().numpy()[0]
    print(W.shape)
    # W =  W.reshape((72,72))
    W =  W.reshape((6, 6, 12, 12))
    h,w,_,_ = W.shape
    

    for i in range(1, h+1):
        print(i)
        for j in range(1, w+1):
            plt.subplot(6,6,(i-1)*6+j)
            plt.imshow(W[i-1, j-1], vmin=-1, vmax=1, cmap='seismic')


    # plt.show()
    f = plt.gcf()  #获取当前图像
    f.savefig('/workspace/home/huangxiaoshuang/medicine/hcc-prognostic/logs/classification/ckpt/1cls_v2.4.17/{}.png'.format("plot_w"))
    f.clear()  #释放内存


if __name__ == '__main__':
    main()