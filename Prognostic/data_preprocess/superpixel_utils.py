# from tiatoolbox.wsicore.wsireader import WSIReader
# from matplotlib import pyplot as plt
# wsi = WSIReader.open(input_img="./CMU-1.ndpi")
# # Read a region at level 0 (baseline / full resolution)
# bounds = [1000, 2000, 2000, 3000]
# img = wsi.read_bounds(bounds)
# plt.imshow(img)
# # This could also be written more verbosely as follows
# img = wsi.read_bounds(
#     bounds,
#     resolution=0,
#     units="level",
# )
# plt.imshow(img)