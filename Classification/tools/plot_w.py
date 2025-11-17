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
    print(ckpt['state_dict']['module.crf.W'].cpu().numpy(), ckpt['state_dict']['module.crf.W'].cpu().numpy().shape)
    W = ckpt['state_dict']['module.crf.W'].cpu().numpy()[0].reshape((3, 3, 3, 3))

    plt.subplot(331)
    plt.imshow(W[0, 0], vmin=-1, vmax=1, cmap='seismic')
    plt.subplot(332)
    plt.imshow(W[0, 1], vmin=-1, vmax=1, cmap='seismic')
    plt.subplot(333)
    plt.imshow(W[0, 2], vmin=-1, vmax=1, cmap='seismic')

    plt.subplot(334)
    plt.imshow(W[1, 0], vmin=-1, vmax=1, cmap='seismic')
    plt.subplot(335)
    plt.imshow(W[1, 1], vmin=-1, vmax=1, cmap='seismic')
    plt.subplot(336)
    plt.imshow(W[1, 2], vmin=-1, vmax=1, cmap='seismic')

    plt.subplot(337)
    plt.imshow(W[2, 0], vmin=-1, vmax=1, cmap='seismic')
    plt.subplot(338)
    plt.imshow(W[2, 1], vmin=-1, vmax=1, cmap='seismic')
    plt.subplot(339)
    plt.imshow(W[2, 2], vmin=-1, vmax=1, cmap='seismic')

    # plt.show()
    f = plt.gcf()  #获取当前图像
    f.savefig('/home/huangxiaoshuang/medicine/HCC_Prognostic/logs/classification/ckpt/1cls_v4.0.2/{}.png'.format("plot_w"))
    f.clear()  #释放内存


if __name__ == '__main__':
    main()