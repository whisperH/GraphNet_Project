import os
import openslide
import matplotlib.pyplot as plt

if __name__ == '__main__':

    root_dir = "/media/whisper/Newsmy/WSI_RawData/CY_SL_FD_HS/prognostic"
    batch_name = ["huashan"]
    slide = openslide.open_slide(os.path.join(root_dir, batch_name[0], "D16-01976-10 1.tif"))
    print(slide.level_dimensions)
    img = slide.read_region((6400, 31744), 0, (256, 256)).convert('RGB')
    # img = slide.get_thumbnail((1637, 3942))
    plt.imshow(img)
    plt.show()