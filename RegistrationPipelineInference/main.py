# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# from ContrastiveUnpairedTranslation import *
from SuperPoint.SuperpointFunctions import *
from ContrastiveUnpairedTranslation import test, util, options, models, data
import time


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    m = test.load_model()
    sp_m = load_tensorflow_model("d:/dev/python/SuperPoint/exper/saved_models/sp_v6/")
    t = test.convert_data_to_tensor(
        "D:/datasets/Image_registration/230123NH-77995rqivU/trainA/p1_wA1_t1_m1010_c0_z1_l1_o0.png")
    r = test.inference(m, t)
    res_img = util.util.tensor2im(r)
    #im = Image.fromarray(i m)
    #im.save("testfake_B.png")
    # define a transform to convert a tensor to PIL image
    img_size = (256, 256)
    # im2 = test.convert_data_to_tensor("D:/datasets/Image_registration/230123NH-77995rqivU/trainB/p1_wA1_t1_m1010_c0_z1_l1_o0.png")
    img1, img1_orig = preprocess_image(res_img, img_size)
    img2, img2_orig = preprocess_image(res_img, img_size)
    res = compute_superpoint(img1,  img2)
    print(res)
    start = time.time()
    r = test.inference(m, t)
    res_img = util.util.tensor2im(r)
    # im = Image.fromarray(i m)
    # im.save("testfake_B.png")
    # define a transform to convert a tensor to PIL image
    img_size = (256, 256)
    # im2 = test.convert_data_to_tensor("D:/datasets/Image_registration/230123NH-77995rqivU/trainB/p1_wA1_t1_m1010_c0_z1_l1_o0.png")
    img1, img1_orig = preprocess_image(res_img, img_size)
    img2, img2_orig = preprocess_image(res_img, img_size)
    res = compute_superpoint(img1, img2)
    end = time.time()
    print(end - start)
    print(res)
