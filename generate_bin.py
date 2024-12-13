##########################ok#############################################################################################
import cv2
import numpy as np
import os
import random
import math
from augmentation_utils import random_hsv, augment_polygon

# Define your folder paths
folder_mask = 'E:/masklablechange/txtlabels/masks'  
folder_A = 'E:/masklablechange/txtlabels/imagepairs/A'  
folder_B = 'E:/masklablechange/txtlabels/imagepairs/B'  

output_A = 'E:/masklablechange/txtlabels/imagepairs/out_new/A'
output_B = 'E:/masklablechange/txtlabels/imagepairs/out_new/B'
output_label = 'E:/masklablechange/txtlabels/imagepairs/out_new/Labels'

os.makedirs(output_A, exist_ok=True)
os.makedirs(output_B, exist_ok=True)
os.makedirs(output_label, exist_ok=True)

# Load images once to reduce I/O operations
images_mask = [cv2.imread(os.path.join(folder_mask, img)) for img in os.listdir(folder_mask) if img.endswith('.png')]
images_A = [cv2.imread(os.path.join(folder_A, img)) for img in os.listdir(folder_A) if img.endswith('.jpg')]
images_B = [cv2.imread(os.path.join(folder_B, img)) for img in os.listdir(folder_B) if img.endswith('.jpg')]

count =6000
for t in range(0,10):
    for filename in os.listdir(folder_A):
        print(t, filename)
        count += 1
        
        # Randomly select masks
        mask = random.choice(images_mask)
        mask_patch = random.choice(images_mask)

        image_A = cv2.imread(os.path.join(folder_A, filename))
        image_B = cv2.imread(os.path.join(folder_B, filename))

        if mask is not None and image_A is not None and image_B is not None:
            mask = cv2.resize(mask, (image_A.shape[1], image_A.shape[0]), interpolation=cv2.INTER_CUBIC)
            mask_patch = cv2.resize(mask_patch, (image_B.shape[1], image_B.shape[0]), interpolation=cv2.INTER_CUBIC)
     
            h, w = mask.shape[:2]
            center_x, center_y = w // 2, h // 2
            num_points = 10  
            radius = 50      
            points = [(int(center_x + radius * math.cos(2 * math.pi * i / num_points)),
                        int(center_y + radius * math.sin(2 * math.pi * i / num_points))) for i in range(num_points)]

            coords = np.array(points, dtype=np.int32)

            # Apply polygon augmentation
            augmented_points = augment_polygon(coords, image_A.shape[1], image_A.shape[0])
            if len(augmented_points) > 0:
                augmented_points = np.array(augmented_points, dtype=np.int32)
                M = cv2.getAffineTransform(coords[:3].astype(np.float32), augmented_points[:3].astype(np.float32))

                transformed_mask_A = cv2.warpAffine(mask, M, (image_A.shape[1], image_A.shape[0]), flags=cv2.INTER_NEAREST)
                transformed_mask_B = cv2.warpAffine(mask, M, (image_B.shape[1], image_B.shape[0]), flags=cv2.INTER_NEAREST)
                transformed_mask_another = cv2.warpAffine(mask_patch, M, (image_B.shape[1], image_B.shape[0]), flags=cv2.INTER_NEAREST)

                # 使用位运算更新 image_A
                mask_A = (transformed_mask_A > 0).any(axis=2).astype(np.uint8)  
                image_A = cv2.bitwise_or(image_A, transformed_mask_A * mask_A[:, :, np.newaxis])  

                aug_A_filename = os.path.join(output_A, str(count) + f'_{filename}')
                cv2.imwrite(aug_A_filename, image_A)  

                # 使用位运算更新 image_B
                mask_B = (transformed_mask_B > 0).any(axis=2).astype(np.uint8)  
                image_B = cv2.bitwise_or(image_B, transformed_mask_B * mask_B[:, :, np.newaxis])  

                mask_another = (transformed_mask_another > 0).any(axis=2).astype(np.uint8)  
                image_B = cv2.bitwise_or(image_B, transformed_mask_another * mask_another[:, :, np.newaxis])  

                aug_B_filename = os.path.join(output_B, str(count) + f'_{filename}')
                cv2.imwrite(aug_B_filename, image_B)  

                # 将 transformed_mask_another 转换为二值图
                gray_mask = cv2.cvtColor(transformed_mask_another, cv2.COLOR_BGR2GRAY)
                _, transformed_mask_another_bin = cv2.threshold(gray_mask, 1, 255, cv2.THRESH_BINARY)

                # 保存二值化的掩码
                auga_bin_filename = os.path.join(output_label, str(count) + f'_{filename}')
                cv2.imwrite(auga_bin_filename.replace(".jpg", ".png"), transformed_mask_another_bin)

                

# import cv2
# import numpy as np
# import os
# import random
# import math
# from augmentation_utils import random_hsv, augment_polygon, get_polygon_centroid

# # Define your folder paths
# folder_mask = 'E:/masklablechange/txtlabels/masks'  # Folder containing masks (as .txt files)
# folder_A = 'E:/masklablechange/txtlabels/imagepairs/A'  # Folder containing images in Folder B
# folder_B = 'E:/masklablechange/txtlabels/imagepairs/B'  # Folder containing images in Folder C

# output_A = 'E:/masklablechange/txtlabels/imagepairs/out/A'
# output_B = 'E:/masklablechange/txtlabels/imagepairs/out/B'
# output_label = 'E:/masklablechange/txtlabels/imagepairs/out/Labels'

# os.makedirs(output_A, exist_ok=True)
# os.makedirs(output_B, exist_ok=True)
# os.makedirs(output_label, exist_ok=True)



# # Get list of images in folders B and C
# images_mask = [os.path.join(folder_mask, img) for img in os.listdir(folder_mask) if img.endswith('.png')]
# images_A = [os.path.join(folder_A, img) for img in os.listdir(folder_A) if img.endswith('.jpg')]
# images_B = [os.path.join(folder_B, img) for img in os.listdir(folder_B) if img.endswith('.jpg')]


# count=0
# for t in range(10):
#     # Traverse mask files in folder A
#     for filename in os.listdir(folder_A):
#             print(t,filename)

#             count=count+1        
#             # 假设 images_mask 已包含所有图片路径
#             selected_images = random.sample(images_mask, 5)

#             mask=cv2.imread(selected_images[0])

#             # 假设 images_mask 已包含所有图片路径
#             selected_images_patch = random.sample(images_mask, 1)
#             mask_patch=cv2.imread(selected_images_patch[0])


#             image_path_A = os.path.join(folder_A, filename) 
#             image_A = cv2.imread(image_path_A)

#             image_path_B = os.path.join(folder_B, filename) 
#             image_B = cv2.imread(image_path_B)


#             if mask is not None and image_A is not None and image_B is not None:
#                 mask = cv2.resize(mask, (image_A.shape[1], image_A.shape[0]), interpolation=cv2.INTER_CUBIC)
            
#             if mask_patch is not None and image_A is not None and image_B is not None:
#                 mask_patch = cv2.resize(mask_patch, (image_B.shape[1], image_B.shape[0]), interpolation=cv2.INTER_CUBIC)
     
#             h, w = mask.shape[:2]
#             center_x, center_y = w // 2, h // 2
#             num_points = 10  # 顶点数量
#             radius = 50      # 点到中心的距离
#             points = []
        
#             # 生成中心多边形的顶点
#             for i in range(num_points):
#                 angle = 2 * math.pi * i / num_points  # 每个点的角度
#                 x = int(center_x + radius * math.cos(angle))
#                 y = int(center_y + radius * math.sin(angle))
#                 points.append((x, y))


#             coords = np.array(points, dtype=np.float32).reshape(-1, 2)
#             coords = np.array(coords, dtype=np.int32)

#             # Apply polygon augmentation
#             augmented_points = augment_polygon(coords, image_A.shape[1], image_A.shape[0])
#             flag=0
#             if len(augmented_points) > 0:
#                 flag = 1
#                 augmented_points = np.array(augmented_points, dtype=np.int32)
#                 M = cv2.getAffineTransform(coords[:3].astype(np.float32), augmented_points[:3].astype(np.float32))
#                 transformed_mask_A = cv2.warpAffine(mask, M, (image_A.shape[1], image_A.shape[0]), flags=cv2.INTER_NEAREST)
#                 transformed_mask_B = cv2.warpAffine(mask, M, (image_B.shape[1], image_B.shape[0]), flags=cv2.INTER_NEAREST)
#                 transformed_mask_another = cv2.warpAffine(mask_patch, M, (image_B.shape[1], image_B.shape[0]), flags=cv2.INTER_NEAREST)

#                 # transformed_mask_A = mask
#                 # transformed_mask_B = mask
#                 # transformed_mask_another =mask_patch


#             if flag:
                
#                 #########################bb###################
#                 image_A[(transformed_mask_A > 0).all(axis=2)]=transformed_mask_A[(transformed_mask_A > 0).all(axis=2)]               
#                 #aug_A_filename = os.path.join(output_A, str(count)+f'_{filename}'.replace(".jpg","_A.jpg"))
#                 aug_A_filename = os.path.join(output_A, str(count)+f'_{filename}')
#                 #cv2.imwrite(aug_A_filename.replace(".jpg", ".png"), image_A)  # 保存更新后的图像
#                 cv2.imwrite(aug_A_filename, image_A)  # 保存更新后的图像


#                 image_B[(transformed_mask_B>0).all(axis=2)]=transformed_mask_B[(transformed_mask_B>0).all(axis=2)]
#                 image_B[(transformed_mask_another>0).all(axis=2)]=transformed_mask_another[(transformed_mask_another>0).all(axis=2)]
#                 aug_B_filename = os.path.join(output_B, str(count)+f'_{filename}')
#                 #cv2.imwrite(aug_B_filename.replace(".jpg", ".png"), image_B)  # 保存更新后的图像
#                 cv2.imwrite(aug_B_filename, image_B)  # 保存更新后的图像

#                 # 将 transformed_mask_another 转换为二值图
#                 # 
#                 gray_mask = cv2.cvtColor(transformed_mask_another, cv2.COLOR_BGR2GRAY)
#                 _, transformed_mask_another_bin = cv2.threshold(gray_mask, 0, 255, cv2.THRESH_BINARY)
#                 # 保存二值化的掩码
#                 auga_bin_filename = os.path.join(output_label, str(count) + f'_{filename}')
#                 cv2.imwrite(auga_bin_filename.replace(".jpg", ".png"), transformed_mask_another_bin)  
                
                
#                 # transformed_mask_another[(transformed_mask_another > 0).all(axis=2)] = [255, 255, 255]
#                 # transformed_mask_another[(transformed_mask_another == 0).all(axis=2)] = [0, 0, 0]
#                 # auga_bin_filename = os.path.join(output_label, str(count)+f'_{filename}')
#                 # cv2.imwrite(auga_bin_filename.replace(".jpg", ".png"), transformed_mask_another)  # 保存更新后的图像


# import cv2
# import numpy as np
# import os
# import random
# from augmentation_utils import random_hsv, augment_polygon, get_polygon_centroid

# # Define your folder paths
# folder_A = 'E:/masklablechange/A'  # Folder containing masks (as .txt files)
# folder_B = 'E:/masklablechange/C'  # Folder containing images

# # Create output directories for saved results
# output_rgb = 'E:/masklablechange/outxxx'  # Folder to save RGB images with masks
# output_binary = 'E:/masklablechange/outxxx'  # Folder to save binary masks
# os.makedirs(output_rgb, exist_ok=True)
# os.makedirs(output_binary, exist_ok=True)

# # Specify the class label to mask (e.g., '1' for class 1)
# target_class_label = '1'  # Adjust according to your needs

# # 遍历文件夹 A 中的遮罩文件
# for mask_filename in os.listdir(folder_A):
#     # 检查文件是否是遮罩（例如，是否以 .txt 结尾）
#     if mask_filename.endswith('.txt'):
#         # 构造文件夹 B 中相应的图像文件名
#         image_filename = mask_filename.replace('.txt', '.jpg')  # 假设图像是 .jpg 格式
#         mask_path = os.path.join(folder_A, mask_filename)
#         image_path_B = os.path.join(folder_B, image_filename)

#         # 检查图像是否存在于文件夹 B
#         if os.path.exists(image_path_B):
#             # 加载文件夹 A 中的原始图像
#             image_path_A = os.path.join(folder_A, image_filename)  # 假设 A 中也有同名图像
#             image_A = cv2.imread(image_path_A)

#             # 加载文件夹 B 中的图像
#             image_B = cv2.imread(image_path_B)

#             # 假设已经有了 image_A 和 binary_mask
#             filled_image_A = np.zeros_like(image_A)
#             binary_mask = np.zeros((image_B.shape[0], image_B.shape[1]), dtype=np.uint8)
#             # 读取遮罩文件并处理坐标
#             with open(mask_path, 'r') as file:
#                 flag=0
#                 for line in file:
#                     parts = line.strip().split()
#                     if len(parts) > 1 and parts[0] == target_class_label:
#                         # 转换为像素坐标
#                         coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
#                         pixel_coords = np.zeros_like(coords)
#                         pixel_coords[:, 0] = coords[:, 0] * image_B.shape[1]  # x 坐标
#                         pixel_coords[:, 1] = coords[:, 1] * image_B.shape[0]  # y 坐标
                        
#                         coords = np.array(pixel_coords, dtype=np.int32)
#                         cv2.fillPoly(binary_mask, [coords], 255)  # 填充遮罩
                        
#                         # 将A中的对应区域的像素值应用到B中
#                         filled_image_A[binary_mask == 255] = image_A[binary_mask == 255]
           
#                         # 应用增强到多边形
#                         augmented_points = augment_polygon(pixel_coords.tolist(), image_A.shape[1], image_A.shape[0])


#                         # 创建增强坐标的多边形
#                         if len(augmented_points) > 0:
#                             flag=1
#                             augmented_points = np.array(augmented_points, dtype=np.int32)

#                             # 对遮罩进行放射变换
#                             # 计算变换矩阵
#                             src_points = np.array(coords, dtype=np.float32)
#                             dst_points = np.array(augmented_points, dtype=np.float32)  # 这里假设不进行变化

#                             # 生成放射变换矩阵
#                             M = cv2.getAffineTransform(src_points[:3], dst_points[:3])
            
#                             # 应用放射变换到遮罩
#                             transformed_mask = cv2.warpAffine(filled_image_A, M, (image_A.shape[1], image_A.shape[0]))

#             if flag:

#                 rgb_ori_filename = os.path.join(output_rgb, f'rgb_ori_{image_filename}')
#                 cv2.imwrite(rgb_ori_filename, image_B)  # 保存更新后的图像

#                 image_B[transformed_mask>0]=transformed_mask[transformed_mask>0]

#                 aug_bin_filename = os.path.join(output_binary, f'masked_aug_{image_filename}')
#                 transformed_mask[transformed_mask>0]=255
#                 cv2.imwrite(aug_bin_filename, transformed_mask)  # 保存更新后的图像

#                 ori_bin_filename = os.path.join(output_binary, f'masked_ori_{image_filename}')
#                 cv2.imwrite(ori_bin_filename, binary_mask)  # 保存更新后的图像


#                 rgb_aug_filename = os.path.join(output_rgb, f'rgb_aug_{image_filename}')
#                 cv2.imwrite(rgb_aug_filename, image_B)  # 保存更新后的图像

#############################################################################################
# import cv2
# import numpy as np
# import os
# import random
# from augmentation_utils import random_hsv, augment_polygon, get_polygon_centroid

# # Define your folder paths
# folder_A = 'E:/masklablechange/txtlabels/tongdaoyinhuan-gy-2'  # Folder containing masks (as .txt files)
# folder_B = 'E:/masklablechange/txtlabels/imagepairs/A'  # Folder containing images in Folder B
# folder_C = 'E:/masklablechange/txtlabels/imagepairs/B'  # Folder containing images in Folder C

# # Create output directories for results in BB and CC
# output_BB = 'E:/masklablechange/txtlabels/imagepairs/out/AA'
# output_CC = 'E:/masklablechange/txtlabels/imagepairs/out/BB'

# # Specify the class label to mask (e.g., '1' for class 1)
# #target_class_label = '1'  # Adjust according to your needs
# target_labels = ['0', '1', '2']  # Specify all possible target class labels

# # output_BB_class = os.path.join(f"{output_BB}_{target_class_label}")
# # output_CC_class = os.path.join(f"{output_CC}_{target_class_label}")

# output_BB_class = output_BB
# output_CC_class = output_CC

# # Ensure directories exist
# os.makedirs(output_BB_class, exist_ok=True)
# os.makedirs(output_CC_class, exist_ok=True)




# # Get list of images in folders B and C
# images_B = [os.path.join(folder_B, img) for img in os.listdir(folder_B) if img.endswith('.jpg')]
# images_C = [os.path.join(folder_C, img) for img in os.listdir(folder_C) if img.endswith('.jpg')]

# # Traverse mask files in folder A
# for mask_filename in os.listdir(folder_A):
#     if mask_filename.endswith('.txt'):
#         mask_path = os.path.join(folder_A, mask_filename)
#         image_path_A = os.path.join(folder_A, mask_path.replace(".txt", ".jpg")) 
#         image_A = cv2.imread(image_path_A)

#         # Randomly select an image from folders B and C
#         for image_filename in os.listdir(folder_B):
#             image_path_B =os.path.join(folder_B, image_filename) 
#             image_path_C = os.path.join(folder_C, image_filename) 

#             # Load images from folders B and C
#             image_B = cv2.imread(image_path_B)
#             image_C = cv2.imread(image_path_C)
                
#             # Resize image_A to match the dimensions of image_B
#             if image_A is not None and image_B is not None:
#                     image_A = cv2.resize(image_A, (image_B.shape[1], image_B.shape[0]))


#             # Create binary mask and filled overlay images
#             binary_mask = np.zeros((image_A.shape[0], image_A.shape[1]), dtype=np.uint8)


#             filled_image_A = np.zeros_like(image_A)
#             #filled_image_B = np.zeros_like(image_B)
#             #filled_image_C = np.zeros_like(image_C)
            
#             # Read mask coordinates
#             with open(mask_path, 'r') as file:
#                 flag = 0
#                 for line in file:
#                     parts = line.strip().split()
#                     print("XXXXXXXXXXXXXX",parts[0])
#                     if len(parts) > 1 and parts[0] in target_labels:#parts[0] == target_class_label:
#                         print("XXXXXXXXXXXXXX")
#                         coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
#                         pixel_coords = np.zeros_like(coords)
#                         pixel_coords[:, 0] = coords[:, 0] * image_B.shape[1]
#                         pixel_coords[:, 1] = coords[:, 1] * image_B.shape[0]
#                         coords = np.array(pixel_coords, dtype=np.int32)
#                         cv2.fillPoly(binary_mask, [coords], 255)

#                         # Copy overlay area from B and C based on the mask
#                         filled_image_A[binary_mask == 255] = image_A[binary_mask == 255]
#                         #filled_image_C[binary_mask == 255] = image_C[binary_mask == 255]

#                         # Apply polygon augmentation
#                         augmented_points = augment_polygon(pixel_coords.tolist(), image_A.shape[1], image_A.shape[0])
#                         if len(augmented_points) > 0:
#                             flag = 1
#                             augmented_points = np.array(augmented_points, dtype=np.int32)
#                             M = cv2.getAffineTransform(coords[:3].astype(np.float32), augmented_points[:3].astype(np.float32))
#                             transformed_mask_B = cv2.warpAffine(filled_image_A, M, (image_B.shape[1], image_B.shape[0]))
#                             transformed_mask_C = cv2.warpAffine(filled_image_A, M, (image_C.shape[1], image_C.shape[0]))

#                 if flag:
#                     #########################bb###################
#                     rgb_ori_filename = os.path.join(output_BB_class, f'rgb_ori_{image_filename}')
#                     cv2.imwrite(rgb_ori_filename, image_B)  # 保存更新后的图像

#                     image_B[transformed_mask_B>0]=transformed_mask_B[transformed_mask_B>0]

#                     aug_bin_filename = os.path.join(output_BB_class, f'masked_aug_{image_filename}')
#                     transformed_mask_B[transformed_mask_B>0]=255
#                     cv2.imwrite(aug_bin_filename, transformed_mask_B)  # 保存更新后的图像

#                     ori_bin_filename = os.path.join(output_BB_class, f'masked_ori_{image_filename}')
#                     cv2.imwrite(ori_bin_filename, binary_mask)  # 保存更新后的图像


#                     rgb_aug_filename = os.path.join(output_BB_class, f'rgb_aug_{image_filename}')
#                     cv2.imwrite(rgb_aug_filename, image_B)  # 保存更新后的图像


#                     #########################CC##################################################################
#                     rgb_ori_filename = os.path.join(output_CC_class, f'rgb_ori_{image_filename}')
#                     cv2.imwrite(rgb_ori_filename, image_C)  # 保存更新后的图像

#                     image_C[transformed_mask_C>0]=transformed_mask_C[transformed_mask_C>0]

#                     aug_bin_filename = os.path.join(output_CC_class, f'masked_aug_{image_filename}')
#                     transformed_mask_C[transformed_mask_C>0]=255
#                     cv2.imwrite(aug_bin_filename, transformed_mask_C)  # 保存更新后的图像

#                     ori_bin_filename = os.path.join(output_CC_class, f'masked_ori_{image_filename}')
#                     cv2.imwrite(ori_bin_filename, binary_mask)  # 保存更新后的图像


#                     rgb_aug_filename = os.path.join(output_CC_class, f'rgb_aug_{image_filename}')
#                     cv2.imwrite(rgb_aug_filename, image_C)  # 保存更新后的图像






                
##################################贴mask######################3
# import cv2
# import numpy as np
# import os
# import random
# import math
# from augmentation_utils import random_hsv, augment_polygon, get_polygon_centroid

# # 初始化路径
# folder_mask = 'E:/masklablechange/txtlabels/masks'
# folder_A = 'E:/masklablechange/txtlabels/imagepairs/A'
# folder_B = 'E:/masklablechange/txtlabels/imagepairs/B'
# output = 'E:/masklablechange/txtlabels/imagepairs/out'
# os.makedirs(output, exist_ok=True)

# # 文件列表
# images_mask = [os.path.join(folder_mask, img) for img in os.listdir(folder_mask) if img.endswith('.jpg')]
# images_A = [os.path.join(folder_A, img) for img in os.listdir(folder_A) if img.endswith('.jpg')]
# images_B = [os.path.join(folder_B, img) for img in os.listdir(folder_B) if img.endswith('.jpg')]

# count = 0
# for t in range(20):
#     for filename in os.listdir(folder_A):
#         count += 1        
#         selected_images = random.sample(images_mask, 5)
#         mask = cv2.imread(selected_images[0])

#         selected_images_patch = random.sample(images_mask, 1)
#         mask_patch = cv2.imread(selected_images_patch[0])

#         image_A = cv2.imread(os.path.join(folder_A, filename))
#         image_B = cv2.imread(os.path.join(folder_B, filename))

#         if mask is not None and image_A is not None:
#             mask = cv2.resize(mask, (image_A.shape[1], image_A.shape[0]))

#         if mask_patch is not None and image_B is not None:
#             mask_patch = cv2.resize(mask_patch, (image_B.shape[1], image_B.shape[0]))

#         h, w = mask.shape[:2]
#         center_x, center_y = w // 2, h // 2
#         num_points = 10
#         radius = 50
#         points = [(int(center_x + radius * math.cos(2 * math.pi * i / num_points)),
#                    int(center_y + radius * math.sin(2 * math.pi * i / num_points)))
#                   for i in range(num_points)]

#         coords = np.array(points, dtype=np.float32).reshape(-1, 2)
#         coords = np.array(coords, dtype=np.int32)

#         # 增强多边形
#         augmented_points = augment_polygon(coords, image_A.shape[1], image_A.shape[0])

#         # 确保 augmented_points 是一个 NumPy 数组
#         augmented_points = np.array(augmented_points, dtype=np.float32)


#         if len(augmented_points) > 0:
#             M = cv2.getAffineTransform(coords[:3].astype(np.float32), augmented_points[:3].astype(np.float32))
#             transformed_mask_A = cv2.warpAffine(mask, M, (image_A.shape[1], image_A.shape[0]))
#             transformed_mask_B = cv2.warpAffine(mask, M, (image_B.shape[1], image_B.shape[0]))
#             transformed_mask_another = cv2.warpAffine(mask_patch, M, (image_B.shape[1], image_B.shape[0]))

#             # 应用二值掩码
#             binary_mask = cv2.threshold(transformed_mask_another, 1, 255, cv2.THRESH_BINARY)[1]
#             kernel = np.ones((3, 3), np.uint8)
#             binary_mask = cv2.erode(binary_mask, kernel, iterations=1)

#             # 图像融合
#             mask_A_nonzero = transformed_mask_A > 0
#             blended_region_A = cv2.addWeighted(image_A, 0.5, transformed_mask_A, 0.5, 0)
#             image_A[mask_A_nonzero] = blended_region_A[mask_A_nonzero]
#             aug_A_filename = os.path.join(output, str(count) + f'_{filename}'.replace(".jpg", "_A.jpg"))
#             cv2.imwrite(aug_A_filename, image_A)

#             mask_B_nonzero = transformed_mask_B > 0
#             blended_region_B = cv2.addWeighted(image_B, 0.5, transformed_mask_B, 0.5, 0)
#             image_B[mask_B_nonzero] = blended_region_B[mask_B_nonzero]

#             mask_another_nonzero = binary_mask > 0
#             blended_region_another = cv2.addWeighted(image_B, 0.5, transformed_mask_another, 0.5, 0)
#             image_B[mask_another_nonzero] = blended_region_another[mask_another_nonzero]

#             aug_B_filename = os.path.join(output, str(count) + f'_{filename}'.replace(".jpg", "_B.jpg"))
#             cv2.imwrite(aug_B_filename, image_B)

#             auga_bin_filename = os.path.join(output, str(count) + f'_{filename}'.replace(".jpg", "_mask.jpg"))
#             cv2.imwrite(auga_bin_filename.replace(".jpg", ".png"), binary_mask)


# ##########################ok#############################################################################################33
# import cv2
# import numpy as np
# import os
# import random
# import math
# from augmentation_utils import random_hsv, augment_polygon, get_polygon_centroid

# # Define your folder paths
# folder_mask = 'E:/masklablechange/txtlabels/masks'  # Folder containing masks (as .txt files)
# folder_A = 'E:/masklablechange/txtlabels/imagepairs/A'  # Folder containing images in Folder B
# folder_B = 'E:/masklablechange/txtlabels/imagepairs/B'  # Folder containing images in Folder C

# output_A = 'E:/masklablechange/txtlabels/imagepairs/out/A'
# output_B = 'E:/masklablechange/txtlabels/imagepairs/out/B'
# output_label = 'E:/masklablechange/txtlabels/imagepairs/out/Labels'

# os.makedirs(output_A, exist_ok=True)
# os.makedirs(output_B, exist_ok=True)
# os.makedirs(output_label, exist_ok=True)



# # Get list of images in folders B and C
# images_mask = [os.path.join(folder_mask, img) for img in os.listdir(folder_mask) if img.endswith('.png')]
# images_A = [os.path.join(folder_A, img) for img in os.listdir(folder_A) if img.endswith('.jpg')]
# images_B = [os.path.join(folder_B, img) for img in os.listdir(folder_B) if img.endswith('.jpg')]


# count=0
# for t in range(20):
#     # Traverse mask files in folder A
#     for filename in os.listdir(folder_A):
#             print(t,filename)

#             count=count+1        
#             # 假设 images_mask 已包含所有图片路径
#             selected_images = random.sample(images_mask, 5)

#             mask=cv2.imread(selected_images[0])

#             # 假设 images_mask 已包含所有图片路径
#             selected_images_patch = random.sample(images_mask, 1)
#             mask_patch=cv2.imread(selected_images_patch[0])


#             image_path_A = os.path.join(folder_A, filename) 
#             image_A = cv2.imread(image_path_A)

#             image_path_B = os.path.join(folder_B, filename) 
#             image_B = cv2.imread(image_path_B)


#             # # 创建一个全零的图像（黑色图像）
#             # # 找到裁剪图像中的非零区域
#             # non_zero_indices = np.where(mask != 0)
#             # if non_zero_indices[0].size > 0:  # 检查是否存在非零区域
#             #     # 获取非零区域的边界框
#             #     x_min, y_min = np.min(non_zero_indices[1]), np.min(non_zero_indices[0])  # 列，行
#             #     x_max, y_max = np.max(non_zero_indices[1]), np.max(non_zero_indices[0])  # 列，行

#             #     # 获取目标区域
#             #     target_region = mask[y_min:y_max + 1, x_min:x_max + 1]
                
#             #     new_image = np.zeros((image_A.shape[0], image_A.shape[1], 3), dtype=np.uint8)  # 如果是灰度图像，使用 (new_size[1], new_size[0]) 
#             #     #cv2.imwrite("maskxx.jpg", new_image)  
#             #     x_offset = (image_A.shape[1]- target_region.shape[1]) // 2
#             #     y_offset = (image_A.shape[0]- target_region.shape[0]) // 2
                    
#             #     # 检查目标区域是否超出新图的尺寸
#             #     if x_offset < 0:
#             #         # 目标区域宽度超出新图宽度，调整目标区域的宽度
#             #         target_region = target_region[:, :image_A.shape[1]]
#             #         x_offset = 0

#             #     if y_offset < 0:
#             #         # 目标区域高度超出新图高度，调整目标区域的高度
#             #         target_region = target_region[:image_A.shape[0], :]
#             #         y_offset = 0

#             #     # 确保目标区域在新图中不超出边界
#             #     if (y_offset + target_region.shape[0]) > image_A.shape[0]:
#             #         target_region = target_region[:image_A.shape[0] - y_offset, :]
                    
#             #     if (x_offset + target_region.shape[1]) > image_A.shape[1]: 
#             #         target_region = target_region[:, :image_A.shape[1] - x_offset]

#             #     new_image[y_offset:y_offset + target_region.shape[0], x_offset:x_offset + target_region.shape[1]] = target_region

#             #     mask=new_image
                
#             # else:
#             #     continue
#             #     # 将目标图像放置到全零图像上
#             #     new_image[y_offset:y_offset + target_region.shape[0], x_offset:x_offset + target_region.shape[1]] = target_region
#             #     new_image[y_offset:y_offset + target_region.shape[0], x_offset:x_offset + target_region.shape[1]] = target_region


#             # non_zero_indices = np.where(mask_patch != 0)
#             # if non_zero_indices[0].size > 0:  # 检查是否存在非零区域
#             #     # 获取非零区域的边界框
#             #     x_min, y_min = np.min(non_zero_indices[1]), np.min(non_zero_indices[0])  # 列，行
#             #     x_max, y_max = np.max(non_zero_indices[1]), np.max(non_zero_indices[0])  # 列，行

#             #     # 获取目标区域
#             #     target_region = mask_patch[y_min:y_max + 1, x_min:x_max + 1]
                
#             #     new_image = np.zeros((image_A.shape[0], image_A.shape[1], 3), dtype=np.uint8)  # 如果是灰度图像，使用 (new_size[1], new_size[0]) 
#             #     x_offset = (image_A.shape[1]- target_region.shape[1]) // 2
#             #     y_offset = (image_A.shape[0]- target_region.shape[0]) // 2
                    
#             #     # 检查目标区域是否超出新图的尺寸
#             #     if x_offset < 0:
#             #         # 目标区域宽度超出新图宽度，调整目标区域的宽度
#             #         target_region = target_region[:, :image_A.shape[1]]
#             #         x_offset = 0

#             #     if y_offset < 0:
#             #         # 目标区域高度超出新图高度，调整目标区域的高度
#             #         target_region = target_region[:image_A.shape[0], :]
#             #         y_offset = 0

#             #     # 确保目标区域在新图中不超出边界
#             #     if (y_offset + target_region.shape[0]) > image_A.shape[0]:
#             #         target_region = target_region[:image_A.shape[0] - y_offset, :]
                    
#             #     if (x_offset + target_region.shape[1]) > image_A.shape[1]: 
#             #         target_region = target_region[:, :image_A.shape[1] - x_offset]

#             #     new_image[y_offset:y_offset + target_region.shape[0], x_offset:x_offset + target_region.shape[1]] = target_region

#             #     mask_patch=new_image
                
#             # else:
#             #     continue
#             #     # 将目标图像放置到全零图像上
#             #     new_image[y_offset:y_offset + target_region.shape[0], x_offset:x_offset + target_region.shape[1]] = target_region
#             #     new_image[y_offset:y_offset + target_region.shape[0], x_offset:x_offset + target_region.shape[1]] = target_region

            



#             if mask is not None and image_A is not None and image_B is not None:
#                 mask = cv2.resize(mask, (image_A.shape[1], image_A.shape[0]), interpolation=cv2.INTER_CUBIC)
            
#             if mask_patch is not None and image_A is not None and image_B is not None:
#                 mask_patch = cv2.resize(mask_patch, (image_B.shape[1], image_B.shape[0]), interpolation=cv2.INTER_CUBIC)

         

#             # _,binary_image = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
#             # cv2.imwrite("./x/mask1.jpg", binary_image)  
#             # _,binary_image = cv2.threshold(mask_patch, 0, 255, cv2.THRESH_BINARY)
#             # cv2.imwrite("./x/mask_p2.jpg", binary_image)  



     
#             h, w = mask.shape[:2]
#             center_x, center_y = w // 2, h // 2
#             num_points = 10  # 顶点数量
#             radius = 50      # 点到中心的距离
#             points = []
        
#             # 生成中心多边形的顶点
#             for i in range(num_points):
#                 angle = 2 * math.pi * i / num_points  # 每个点的角度
#                 x = int(center_x + radius * math.cos(angle))
#                 y = int(center_y + radius * math.sin(angle))
#                 points.append((x, y))


#             coords = np.array(points, dtype=np.float32).reshape(-1, 2)
#             coords = np.array(coords, dtype=np.int32)

#             # Apply polygon augmentation
#             augmented_points = augment_polygon(coords, image_A.shape[1], image_A.shape[0])
#             flag=0
#             if len(augmented_points) > 0:
#                 flag = 1
#                 augmented_points = np.array(augmented_points, dtype=np.int32)
#                 M = cv2.getAffineTransform(coords[:3].astype(np.float32), augmented_points[:3].astype(np.float32))
#                 # transformed_mask_A = cv2.warpAffine(mask, M, (image_A.shape[1], image_A.shape[0]), flags=cv2.INTER_NEAREST)
#                 # transformed_mask_B = cv2.warpAffine(mask, M, (image_B.shape[1], image_B.shape[0]), flags=cv2.INTER_NEAREST)
#                 # transformed_mask_another = cv2.warpAffine(mask_patch, M, (image_B.shape[1], image_B.shape[0]), flags=cv2.INTER_NEAREST)

#                 transformed_mask_A = mask
#                 transformed_mask_B = mask
#                 transformed_mask_another =mask_patch


#             if flag:
                
#                 #########################bb###################
#                 image_A[(transformed_mask_A > 0).all(axis=2)]=transformed_mask_A[(transformed_mask_A > 0).all(axis=2)]               
#                 #aug_A_filename = os.path.join(output_A, str(count)+f'_{filename}'.replace(".jpg","_A.jpg"))
#                 aug_A_filename = os.path.join(output_A, str(count)+f'_{filename}')
#                 cv2.imwrite(aug_A_filename, image_A)  # 保存更新后的图像


#                 image_B[(transformed_mask_B>0).all(axis=2)]=transformed_mask_B[(transformed_mask_B>0).all(axis=2)]
#                 image_B[(transformed_mask_another>0).all(axis=2)]=transformed_mask_another[(transformed_mask_another>0).all(axis=2)]
#                 aug_B_filename = os.path.join(output_B, str(count)+f'_{filename}')
#                 cv2.imwrite(aug_B_filename, image_B)  # 保存更新后的图像


#                 transformed_mask_another[(transformed_mask_another > 0).all(axis=2)] = [255, 255, 255]
#                 transformed_mask_another[(transformed_mask_another == 0).all(axis=2)] = [0, 0, 0]
#                 auga_bin_filename = os.path.join(output_label, str(count)+f'_{filename}')
#                 cv2.imwrite(auga_bin_filename.replace(".jpg", ".png"), transformed_mask_another)  # 保存更新后的图像



            

# import cv2
# import numpy as np
# import os
# import random
# from augmentation_utils import random_hsv, augment_polygon, get_polygon_centroid

# # Define your folder paths
# folder_A = 'E:/masklablechange/txtlabels/tongdaoyinhuan-gy-2'  # Folder containing masks (as .txt files)
# folder_B = 'E:/masklablechange/txtlabels/imagepairs/A'  # Folder containing images in Folder B
# folder_C = 'E:/masklablechange/txtlabels/imagepairs/B'  # Folder containing images in Folder C

# # Create output directories for results in BB and CC
# output_BB = 'E:/masklablechange/txtlabels/imagepairs/out/AA'
# output_CC = 'E:/masklablechange/txtlabels/imagepairs/out/BB'

# # Specify the class label to mask (e.g., '1' for class 1)
# #target_class_label = '1'  # Adjust according to your needs
# target_labels = ['0', '1', '2','3','4','5','6','7','8','9','10','11']  # Specify all possible target class labels

# # output_BB_class = os.path.join(f"{output_BB}_{target_class_label}")
# # output_CC_class = os.path.join(f"{output_CC}_{target_class_label}")

# output_BB_class = output_BB
# output_CC_class = output_CC

# # Ensure directories exist
# os.makedirs(output_BB_class, exist_ok=True)
# os.makedirs(output_CC_class, exist_ok=True)




# # Get list of images in folders B and C
# images_B = [os.path.join(folder_B, img) for img in os.listdir(folder_B) if img.endswith('.jpg')]
# images_C = [os.path.join(folder_C, img) for img in os.listdir(folder_C) if img.endswith('.jpg')]


# count=0
# # Traverse mask files in folder A
# for mask_filename in os.listdir(folder_A):
#     if mask_filename.endswith('.txt'):
#         count=count+1
#         print("xxxxxxxxxxxxxxxxxxxxx",count)
#         mask_path = os.path.join(folder_A, mask_filename)
#         image_path_A = os.path.join(folder_A, mask_path.replace(".txt", ".jpg")) 
#         image_A = cv2.imread(image_path_A)

#         # Randomly select an image from folders B and C
#         for image_filename in os.listdir(folder_B):
#             image_path_B =os.path.join(folder_B, image_filename) 
#             image_path_C = os.path.join(folder_C, image_filename) 

#             # Load images from folders B and C
#             image_B = cv2.imread(image_path_B)
#             image_C = cv2.imread(image_path_C)
                
#             # Resize image_A to match the dimensions of image_B
#             if image_A is not None and image_B is not None:
#                     image_A = cv2.resize(image_A, (image_B.shape[1], image_B.shape[0]))


#             # Create binary mask and filled overlay images
#             binary_mask = np.zeros((image_A.shape[0], image_A.shape[1]), dtype=np.uint8)


#             filled_image_A = np.zeros_like(image_A)
#             #filled_image_B = np.zeros_like(image_B)
#             #filled_image_C = np.zeros_like(image_C)
            
#             # Read mask coordinates
#             with open(mask_path, 'r') as file:
#                 flag = 0
#                 for line in file:
#                     parts = line.strip().split()
#                     #print("XXXXXXXXXXXXXX",parts[0])
#                     if len(parts) > 1 and parts[0] in target_labels:#parts[0] == target_class_label:
#                         #print("XXXXXXXXXXXXXX")
#                         coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
#                         pixel_coords = np.zeros_like(coords)
#                         pixel_coords[:, 0] = coords[:, 0] * image_B.shape[1]
#                         pixel_coords[:, 1] = coords[:, 1] * image_B.shape[0]
#                         coords = np.array(pixel_coords, dtype=np.int32)
#                         cv2.fillPoly(binary_mask, [coords], 255)

#                         # Copy overlay area from B and C based on the mask
#                         filled_image_A[binary_mask == 255] = image_A[binary_mask == 255]
#                         #filled_image_C[binary_mask == 255] = image_C[binary_mask == 255]

#                         # Apply polygon augmentation
#                         augmented_points = augment_polygon(pixel_coords.tolist(), image_A.shape[1], image_A.shape[0])
#                         if len(augmented_points) > 0:
#                             flag = 1
#                             augmented_points = np.array(augmented_points, dtype=np.int32)
#                             M = cv2.getAffineTransform(coords[:3].astype(np.float32), augmented_points[:3].astype(np.float32))
#                             transformed_mask_B = cv2.warpAffine(filled_image_A, M, (image_B.shape[1], image_B.shape[0]))
#                             transformed_mask_C = cv2.warpAffine(filled_image_A, M, (image_C.shape[1], image_C.shape[0]))

#                 if flag:
#                     #########################bb###################
#                     #rgb_ori_filename = os.path.join(output_BB_class, str(count)+f'_rgb_ori_{image_filename}')
#                     #cv2.imwrite(rgb_ori_filename, image_B)  # 保存更新后的图像

#                     image_B[transformed_mask_B>0]=transformed_mask_B[transformed_mask_B>0]

#                     aug_bin_filename = os.path.join(output_BB_class, str(count)+f'_masked_aug_{image_filename}')
#                     transformed_mask_B[transformed_mask_B>0]=255
#                     cv2.imwrite(aug_bin_filename.replace(".jpg", ".png"), transformed_mask_B)  # 保存更新后的图像

#                     #ori_bin_filename = os.path.join(output_BB_class, str(count)+f'_masked_ori_{image_filename}')
#                     #cv2.imwrite(ori_bin_filename.replace(".jpg", ".png"), binary_mask)  # 保存更新后的图像


#                     rgb_aug_filename = os.path.join(output_BB_class, str(count)+f'_rgb_aug_{image_filename}')
#                     cv2.imwrite(rgb_aug_filename, image_B)  # 保存更新后的图像


#                     #########################CC##################################################################
#                     #rgb_ori_filename = os.path.join(output_CC_class, str(count)+f'_rgb_ori_{image_filename}')
#                     #cv2.imwrite(rgb_ori_filename, image_C)  # 保存更新后的图像

#                     image_C[transformed_mask_C>0]=transformed_mask_C[transformed_mask_C>0]

#                     aug_bin_filename = os.path.join(output_CC_class, str(count)+f'_masked_aug_{image_filename}')
#                     transformed_mask_C[transformed_mask_C>0]=255
#                     cv2.imwrite(aug_bin_filename.replace(".jpg", ".png"), transformed_mask_C)  # 保存更新后的图像

#                     #ori_bin_filename = os.path.join(output_CC_class, str(count)+f'_masked_ori_{image_filename}')
#                     #cv2.imwrite(ori_bin_filename.replace(".jpg", ".png"), binary_mask)  # 保存更新后的图像


#                     rgb_aug_filename = os.path.join(output_CC_class, str(count)+f'_rgb_aug_{image_filename}')
#                     cv2.imwrite(rgb_aug_filename, image_C)  # 保存更新后的图像
