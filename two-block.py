import os
from datetime import datetime
import cv2

# 指定目录路径
directory_path = r'F:\zhuanyi\cv'

# 获取目录下所有文件的列表
file_list = os.listdir(directory_path)

# 按照文件的修改时间进行排序
file_list.sort(key=lambda x: os.path.getmtime(os.path.join(directory_path, x)))
i=0
# 遍历排序后的文件列表
for file_name in file_list:
    file_path = os.path.join(directory_path, file_name)

    # 获取文件的修改时间
    modified_time = datetime.fromtimestamp(os.path.getmtime(file_path))
    print(file_path)
    print(f"文件名：{file_name}，修改时间：{modified_time}")

    # 打开视频文件
    cap = cv2.VideoCapture(file_path)

    # 初始化变量
    prev_frame_gray_1 = None
    prev_frame_gray_2 = None
    page_images = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # 提取两个感兴趣区域
        roi_1 = frame[1020:1080, 1818:1920]
        roi_2 = frame[400:560, 460:600]

        # 将感兴趣区域转换为灰度图像
        roi_gray_1 = cv2.cvtColor(roi_1, cv2.COLOR_BGR2GRAY)
        roi_gray_2 = cv2.cvtColor(roi_2, cv2.COLOR_BGR2GRAY)

        # 初始化前一帧灰度图像
        if prev_frame_gray_1 is None:
            prev_frame_gray_1 = roi_gray_1.copy()
            prev_frame_gray_2 = roi_gray_2.copy()
            continue

        # 计算当前帧和前一帧的差分图像
        frame_diff_1 = cv2.absdiff(roi_gray_1, prev_frame_gray_1)
        frame_diff_2 = cv2.absdiff(roi_gray_2, prev_frame_gray_2)

        # 对差分图像进行阈值处理
        _, threshold_1 = cv2.threshold(frame_diff_1, 30, 255, cv2.THRESH_BINARY)
        _, threshold_2 = cv2.threshold(frame_diff_2, 30, 255, cv2.THRESH_BINARY)

        # 统计阈值图像中非零像素的数量
        diff_pixels_1 = cv2.countNonZero(threshold_1)
        diff_pixels_2 = cv2.countNonZero(threshold_2)

        # 如果两个区域的差分像素数量都超过阈值，则认为帧发生了变化
        if (diff_pixels_1 > 100 and diff_pixels_2 > 300) or diff_pixels_1>150 or diff_pixels_2>1000:
            print("帧发生变化",i+1,diff_pixels_1,diff_pixels_2)
            page_path = 'output_directory' + "page_" + str(i + 1) + ".jpg"
            cv2.imwrite(page_path, frame)
            page_images.append(frame)
            i += 1

        # 更新前一帧
        prev_frame_gray_1 = roi_gray_1.copy()
        prev_frame_gray_2 = roi_gray_2.copy()

        # 显示当前帧
        cv2.imshow("Frame", frame)

        # 按下 'q' 键退出循环
        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 计算目标帧的索引（比如要跳1000帧）
        target_frame = current_frame + 500
        if target_frame >= max_frames:
            break
        # 跳到目标帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

