from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import cv2
import numpy as np
import os
import re
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def main():

    '''矩形化相关'''
    # folder_path1 = r'D:\dataset\DIR-D\testing\ours'
    # image_files1 = [f for f in os.listdir(folder_path1) if f.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]
    #
    # folder_path2 = r'D:\dataset\DIR-D\testing\gt'
    # image_files2 = [f for f in os.listdir(folder_path2) if f.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]
    #
    # # missing_images = []
    # #
    # # for image in image_files2:
    # #     if image not in image_files1:
    # #         missing_images.append(image)
    # #
    # # # 打印缺失的图片列表
    # # for image in missing_images:
    # #     print(f'Missing Image: {image}')
    #
    # psnr_all = []
    # ssim_all = []
    # for file1 in image_files1:
    #     image1 = cv2.imread(os.path.join(folder_path1, file1))
    #     image2 = cv2.imread(os.path.join(folder_path2, file1))
    #     print(file1)
    #     image1 = cv2.resize(image1,(256,256))
    #     image2 = cv2.resize(image2, (256, 256))
    #     psnr = peak_signal_noise_ratio(image2,image1)
    #     ssim = structural_similarity(image2,image1, multichannel=True)
    #     psnr_all.append(psnr)
    #     ssim_all.append(ssim)
    #
    # average_psnr = np.mean(psnr_all)
    # average_ssim = np.mean(ssim_all)
    #
    # psnr_sort = sorted(psnr_all,reverse=True)
    # ssim_sort = sorted(ssim_all,reverse=True)
    # psnr_30 = sum(psnr_sort[:155]) /155
    # psnr_60 = sum(psnr_sort[155:310]) /155
    # psnr_100 = sum(psnr_sort[310:]) /(len(psnr_sort)-310)
    #
    # ssim_30 = sum(ssim_sort[:155]) /155
    # ssim_60 = sum(ssim_sort[155:310]) /155
    # ssim_100 = sum(ssim_sort[310:]) /(len(psnr_sort)-310)
    #
    # print(f'0-30 PSNR: {psnr_30}')
    # print(f'30-60 PSNR: {psnr_60}')
    # print(f'60-100 PSNR: {psnr_100}')
    # print(f'Average PSNR: {average_psnr}')
    #
    # print(f'0-30 SSIM: {ssim_30}')
    # print(f'30-60 SSIM: {ssim_60}')
    # print(f'60-100 SSIM: {ssim_100}')
    #
    # print(f'Average SSIM: {average_ssim}')

    '''画mse曲线'''

    # data = np.load('D:\\local\\loss2.npy')
    # a= np.mean(data)
    # print(a)
    # np.savetxt('D:\\local\\loss2.txt',data,fmt='%.6f',newline='\n')
    #
    # IHN = np.loadtxt('D:\\IHN-master-yuanban\\IHN-master\\mace.txt')
    # mine = np.loadtxt('D:\\IHN-master-concat-2\\IHN-master\\mace1.txt')
    # dhn = np.loadtxt('D:\\IHN-1\\mace\\dhn_mace2017.txt')
    # udhn = np.loadtxt('D:\\IHN-1\\mace\\udhn_mace2017.txt')
    # dlkfm = np.loadtxt('D:\\dataset\\coco_test\\resultdlkfm.txt')
    # local = np.loadtxt('D:\\local\\loss_orl\\loss2.txt')
    # sift_ransac = np.loadtxt('D:\\IHN-1\\mace\\sift_ransac.txt')
    # sift_magsac = np.loadtxt('D:\\IHN-1\\mace\\sift_magsac.txt')
    # sp_ransac = np.loadtxt('D:\\IHN-1\\mace\\sp_ransac.txt')
    # sp_magsac = np.loadtxt('D:\\IHN-1\\mace\\sp_magsac.txt')
    #
    # fig, ax = plt.subplots()
    # ax.set_xscale("log")
    # ax.grid(True, 'both')
    #
    # a = 1e-2
    # x = []
    # yIHN = []
    # ymine = []
    # ydhn = []
    # yudhn = []
    # ydlkfm = []
    # ylocal = []
    # ysift_ransac = []
    # ysift_magsac = []
    # ysp_ransac = []
    # ysp_magsac = []
    # while True:
    #     for i in range(1, 10):
    #         x.append(a*i)
    #
    #         yIHN.append(np.sum(IHN < a*i) / len(IHN))
    #         ymine.append(np.sum(mine < a * i) / len(mine))
    #         ydhn.append(np.sum(dhn < a * i) / len(dhn))
    #         yudhn.append(np.sum(udhn < a * i) / len(udhn))
    #         ydlkfm.append(np.sum(dlkfm < a * i) / len(dlkfm))
    #         ylocal.append(np.sum(local < a * i) / len(local))
    #         ysift_ransac.append(np.sum(sift_ransac < a * i) / len(sift_ransac))
    #         ysift_magsac.append(np.sum(sift_magsac < a * i) / len(sift_magsac))
    #         ysp_ransac.append(np.sum(sp_ransac < a * i) / len(sp_ransac))
    #         ysp_magsac.append(np.sum(sp_magsac < a * i) / len(sp_magsac))
    #
    #     a *= 10
    #     if a >= 100:
    #         break
    # # print(y)
    # ax.plot(x, yIHN,label = 'IHN')
    # ax.plot(x,ymine, label = 'ours')
    # ax.plot(x,ydhn, label = 'DHN')
    # ax.plot(x, yudhn, label='UDHN')
    # ax.plot(x, ydlkfm, label='DLKFM')
    # ax.plot(x, ylocal, label='LocalTrans')
    # ax.plot(x, ysift_ransac, label='SIFT+RANSAC')
    # ax.plot(x, ysift_magsac, label='SIFT+MAGSAC++')
    # ax.plot(x, ysp_ransac, label='SuperPoint+RANSAC')
    # ax.plot(x, ysp_magsac, label='SuperPoint+MAGSAC++')
    #
    #
    #
    #
    # plt.legend(loc='lower right')
    # plt.xlabel("Average corner error (in pixels)",fontsize=22)
    # plt.ylabel("Fraction of the number of images",fontsize=22)
    # plt.savefig('mace_results.png',dpi=1000)

    '''可视化'''

    img3 = cv2.imread("C:\\Users\\ASUS\\Desktop\\maxMACE\\RealImg\\zfx\\input1\\3.jpg")
    img3 = cv2.resize(img3, (192, 192))
    osp1 = np.array([[32, 32], [32, 160],  [160, 160],[160, 32]], dtype=np.float32)
    # osp1 = np.loadtxt(r'C:\Users\ASUS\Desktop\maxMACE\finalImage\1\label1.txt',dtype=int)
    # flow = np.array([[ 0.6718 , -4.0977], [15.3474 ,-16.7022], [17.7099 , -18.6744], [-5.1347 , 9.6108]], dtype=np.float32)
    flow = np.array([[17.4456,  4.8140], [38.4200,  7.2809], [12.3105, -9.4665], [15.0353, -17.2843]],dtype=np.float32)

    # dst1 = np.array([[154, 14], [138, 157], [259, 162], [257, 38]], dtype=np.float32)
    # dst1= np.loadtxt(r'C:\Users\ASUS\Desktop\maxMACE\finalImage\1\label2.txt',dtype=int)
    dst1 = osp1+flow
    H, _ = cv2.findHomography(osp1, dst1)
    H_inv = np.linalg.inv(H)
    transformed_img = cv2.warpPerspective(img3, H, (img3.shape[1], img3.shape[0]))
    # Display and save the transformed image
    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(transformed_img, cv2.COLOR_BGR2RGB))
    #Optionally, visualize the transformed points on the image
    # polygon1 = Polygon(osp1, closed=True, facecolor='none', edgecolor='green', linewidth=2)
    # polygon2 = Polygon(dst1, closed=True, facecolor='none', edgecolor='red', linewidth=2)
    # ax.add_patch(polygon1)
    # ax.add_patch(polygon2)
    ax.axis('off')
    save_path = 'C:/Users/ASUS/Desktop/maxMACE/RealImg/zfx/input2/warpimg3.jpg'  # Replace with your desired path and filename
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.show()

    #
    # flow1 = np.array([[0.0292, -0.8245], [-1.0849, -3.2211], [-0.0285, -2.3600], [5.3786, -1.3900]], dtype=np.float32)
    #
    # flow_gt = dst1-osp1
    # print('flow_gt:', flow_gt)
    # dst_pred = osp1 + flow
    # mse = np.linalg.norm(flow-flow_gt, axis=1)
    # # mse = np.linalg.norm(flow1, axis=1)
    # print(mse.mean())
    #
    # fig, ax = plt.subplots()
    # ax.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
    # # ax.imshow(img3)
    # # polygon1 = Polygon(osp1, closed=True, facecolor='none', edgecolor='red', linewidth=2)
    # polygon1 = Polygon(dst_pred, closed=True, facecolor='none', edgecolor='red', linewidth=1.5)
    # polygon2 = Polygon(dst1, closed=True, facecolor='none', edgecolor='green', linewidth=2)
    #
    # ax.add_patch(polygon2)
    # ax.add_patch(polygon1)
    # ax.axis('off')
    #
    # save_path = 'C:/Users/ASUS/Desktop/maxMACE/finalImage/1/IHN.jpg'  # Replace with your desired path and filename
    # plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    # plt.show()

    '''输出mse'''
    # dataIHN = np.loadtxt(r'E:\IHN\IHN-master-yuanban\IHN-master\mace.txt')
    # dataRHWF = np.loadtxt(r'D:\RHWF-master\RHWF-master\mace.txt')
    # top_indicesI = np.argsort(dataIHN)[-30:]
    # top_valuesI = dataIHN[top_indicesI]
    # # top_indicesR = np.argsort(dataRHWF)[-30:]
    # # top_valuesR = dataRHWF[top_indicesR]
    #
    # print("最大的20个数:", top_valuesI)
    # print("对应的索引:", top_indicesI)
    #
    # data = np.loadtxt(r'E:\dataset-edge\linshi\label1\413.txt')
    # print(data)

    '''提取边缘'''
    # img1 = cv2.imread('D:\\dataset\\val2017\\input1\\0.jpg')
    # img2 = cv2.imread('D:\\dataset\\val2017\\input2\\0.jpg')
    # img1_gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    # img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # #
    # edges1 = cv2.Canny(image=img1_gray, threshold1=0, threshold2=100)
    # edges1 = cv2.resize(edges1,(256,256))
    # edges2 = cv2.Canny(image=img2_gray, threshold1=0, threshold2=100)
    # edges2 = cv2.resize(edges2, (256, 256))
    # cv2.imshow("img", edges1)
    # cv2.imshow("img2", edges2)
    # cv2.imwrite('D:\\dataset\\val2017\\edge1.jpg',edges1)
    # cv2.imwrite('D:\\dataset\\val2017\\edge2.jpg',edges2)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # img = cv2.imread('E:\\dataset-edge\\HED-BSDS\\HED-BSDS\\test1\\29030.jpg')
    # grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # '''sobel'''
    #
    # x = cv2.Sobel(grayImage, cv2.CV_16S, 1, 0)  # 对x求一阶导
    # y = cv2.Sobel(grayImage, cv2.CV_16S, 0, 1)  # 对y求一阶导
    # absX = cv2.convertScaleAbs(x)
    # absY = cv2.convertScaleAbs(y)
    # Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    '''laplacian'''
    # dst = cv2.Laplacian(grayImage, cv2.CV_16S, ksize=3)
    # Laplacian = cv2.convertScaleAbs(dst)

    # edges1 = cv2.Canny(image=grayImage, threshold1=50, threshold2=150)

    # cv2.imshow("img", Sobel)
    # # cv2.imwrite('D:\\dataset\\val2017\\c1.jpg',edges1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # with open

    # dhn = np.loadtxt('D:\\IHN-1\\mace\\dhn_mace2017.txt')
    # udhn = np.loadtxt('D:\\IHN-1\\mace\\udhn_mace2017.txt')
    # dlkfm = np.loadtxt('D:\\dataset\\coco_test\\resultdlkfm.txt')
    # local = np.loadtxt('D:\\local\\loss_orl\\loss2.txt')
    # IHN = np.loadtxt('D:\\IHN-master-yuanban\\IHN-master\\mace.txt')
    # ours = np.loadtxt('D:\\IHN-master-concat-2\\IHN-master\\test2017mace.txt')
    # sift_ransac = np.loadtxt('D:\\IHN-1\\mace\\sift_ransac.txt')
    # sift_magsac = np.loadtxt('D:\\IHN-1\\mace\\sift_magsac.txt')
    # sp_ransac = np.loadtxt('D:\\IHN-1\\mace\\sp_ransac.txt')
    # sp_magsac = np.loadtxt('D:\\IHN-1\\mace\\sp_magsac.txt')
    # length = len(sp_magsac)
    # ta = int(np.floor(length*0.3))
    # tb = int(np.floor(length*0.6))
    # sp_magsac.sort()
    # print(sp_magsac[0:ta-1].mean())
    # print(sp_magsac[ta:tb-1].mean())
    # print(sp_magsac[tb:length-1].mean())
    # print(udhn.mean())
    # print(dlkfm.mean())
    # print(local.mean())
    # print(sift_ransac.mean())
    # print(sift_magsac.mean())
    # print(sp_ransac.mean())
    # print(sp_magsac.mean())

    # print(IHN.mean())








if __name__ == '__main__':
    main()
