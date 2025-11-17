import numpy as np

np.random.seed(0)

from model.resnet import (resnet18, resnet34, resnet50, resnet101,
                                         resnet152)
# from Classification.model.convnext import (convnext_tiny, convnext_small, convnext_base,
#                                         convnext_large, convnext_xlarge)
# from Classification.model.resnet_dynamic import (resnet18_dynamic, resnet34_dynamic)
# from Classification.model.inceptionv3 import (GoogLeNetv3)
# from Classification.model.inceptionv4 import (inceptionv4)
# from Classification.model.MobileVit import (mobile_vit_xx_small, mobile_vit_x_small, mobile_vit_small)
# from Classification.model.EPatcher import (EPatcher_xx_small)
# from Classification.model.EPatcher_v3 import (EPatcher_xx_small_v3_fast, EPatcher_x_small_v3_fast, EPatcher_small_v3_fast, EPatcher_xx_small_v3)
# from Classification.model.EPatcher_v4 import (EPatcher_xx_small_v4_fast, EPatcher_xx_small_v4_v1, EPatcher_x_small_v4_fast,EPatcher_x_small_v4_v1, EPatcher_small_v4_fast, EPatcher_xx_small_v4, EPatcher_xx_small_v4_fast_rate02_05_10, EPatcher_xx_small_v4_fast_rate01_02_10, EPatcher_x_small_v4, EPatcher_xx_small_v4_v1_rate05_05_10)

MODELS = {'resnet18': resnet18,
          'resnet34': resnet34,
          'resnet50': resnet50,
          'resnet101': resnet101,
          'resnet152': resnet152,
          # 'convnext_tiny': convnext_tiny,
          # 'resnet18_dynamic': resnet18_dynamic,
          # 'resnet34_dynamic': resnet34_dynamic,
          # "GoogLeNetv3": GoogLeNetv3,
          # "GoogLeNetv4": inceptionv4,
          # "mobile_vit_xx_small": mobile_vit_xx_small,
          # "mobile_vit_x_small": mobile_vit_x_small,
          # "mobile_vit_small": mobile_vit_small,
          # "EPatcher_xx_small": EPatcher_xx_small,
          # "EPatcher_xx_small_v3_fast": EPatcher_xx_small_v3_fast,
          # "EPatcher_xx_small_v3": EPatcher_xx_small_v3,
          # "EPatcher_x_small_v3_fast": EPatcher_x_small_v3_fast,
          # "EPatcher_small_v3_fast":EPatcher_small_v3_fast,
          # "EPatcher_xx_small_v4_fast": EPatcher_xx_small_v4_fast,
          # "EPatcher_xx_small_v4": EPatcher_xx_small_v4,
          # "EPatcher_xx_small_v4_v1": EPatcher_xx_small_v4_v1,
          # "EPatcher_x_small_v4_fast": EPatcher_x_small_v4_fast,
          # "EPatcher_xx_small_v4_fast_rate02_05_10": EPatcher_xx_small_v4_fast_rate02_05_10,
          # "EPatcher_xx_small_v4_fast_rate01_02_10": EPatcher_xx_small_v4_fast_rate01_02_10,
          # "EPatcher_x_small_v4_v1": EPatcher_x_small_v4_v1,
          # "EPatcher_x_small_v4": EPatcher_x_small_v4,
          # "EPatcher_xx_small_v4_v1_rate05_05_10": EPatcher_xx_small_v4_v1_rate05_05_10
          }
