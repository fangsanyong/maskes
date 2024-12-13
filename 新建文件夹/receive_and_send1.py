# import cv2
# import subprocess
# import logging

# def stream_rtsp_with_rectangle(rtsp_url, rtsp_target_url):
#     """
#     从 RTSP 流中获取视频，绘制矩形框，并推送到另一个 RTSP 地址
#     :param rtsp_url: 输入 RTSP 流的 URL
#     :param rtsp_target_url: 输出 RTSP 流的目标 URL
#     """
#     try:
#         # 启动 FFmpeg 以推送视频流
#         ffmpeg_command = [
#             'ffmpeg',
#             '-i', rtsp_url,  # 从 RTSP 输入流读取
#             '-vf', 'drawbox=x=(iw/2)-(100/2):y=(ih/2)-(100/2):w=100:h=100:color=red:t=fill',  # 在画面中间绘制红色矩形
#             '-c:v', 'libx264',  # 编码器
#             '-f', 'rtsp',  # 输出格式为 RTSP
#             rtsp_target_url  # RTSP 目标 URL
#         ]
        
#         logging.info(f"推送视频流命令: {' '.join(ffmpeg_command)}")
        
#         # 启动 FFmpeg 子进程
#         process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

#         # 等待 FFmpeg 进程结束
#         process.wait()

#     except Exception as e:
#         logging.error(f"发生错误: {e}")

# # 使用示例
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.DEBUG)
#     rtsp_input_url = "rtsp://10.10.11.16:8556/live.sdp"  # 替换为输入 RTSP 流的 URL
#     rtsp_target_url = "rtsp://10.10.11.16:8557/live.sdp"  # 替换为目标 PC 的 IP 地址
#     stream_rtsp_with_rectangle(rtsp_input_url, rtsp_target_url)



# import cv2
# import subprocess
# import logging

# def stream_rtsp_with_rectangle(rtsp_url, rtsp_target_url):
#     """
#     从 RTSP 流中获取视频，绘制矩形框，并推送到另一个 RTSP 地址
#     :param rtsp_url: 输入 RTSP 流的 URL
#     :param rtsp_target_url: 输出 RTSP 流的目标 URL
#     """
#     try:
#         # 启动 FFmpeg 以推送视频流
#         ffmpeg_command = [
#             'ffmpeg',
#             '-i', rtsp_url,  # 从 RTSP 输入流读取
#             '-vf', 'drawbox=x=(iw/2)-(100/2):y=(ih/2)-(100/2):w=100:h=100:color=red:t=fill',  # 在画面中间绘制红色矩形
#             '-c:v', 'libx264',  # 编码器
#             '-f', 'rtsp',  # 输出格式为 RTSP
#             rtsp_target_url  # RTSP 目标 URL
#         ]
        
#         logging.info(f"推送视频流命令: {' '.join(ffmpeg_command)}")
        
#         # 启动 FFmpeg 子进程
#         process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

#         # 读取 FFmpeg 的输出和错误信息
#         while True:
#             output = process.stdout.readline()
#             error = process.stderr.readline()
#             if output == b'' and process.poll() is not None:
#                 break
#             if output:
#                 logging.info(output.decode().strip())
#             if error:
#                 logging.error(error.decode().strip())

#         # 等待 FFmpeg 进程结束
#         process.wait()

#     except Exception as e:
#         logging.error(f"发生错误: {e}")

# # 使用示例
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.DEBUG)
#     rtsp_input_url = "rtsp://10.10.11.16:8556/live.sdp"  # 替换为输入 RTSP 流的 URL
#     rtsp_target_url = "rtsp://10.10.11.16:8557/live.sdp"  # 替换为目标 PC 的 IP 地址
#     stream_rtsp_with_rectangle(rtsp_input_url, rtsp_target_url)

################################################################################################################################

# import cv2
# import subprocess
# import logging
# import numpy as np

# import os
# os.environ['DISPLAY'] = '10.10.11.16:0'

# def stream_rtsp_with_rectangle(rtsp_url, rtsp_target_url):
#     """
#     从 RTSP 流中获取视频，绘制矩形框，并推送到另一个 RTSP 地址
#     :param rtsp_url: 输入 RTSP 流的 URL
#     :param rtsp_target_url: 输出 RTSP 流的目标 URL
#     """
#     try:
#         ffmpeg_command = [
#             'ffmpeg',
#             '-i', rtsp_url,  # 直接输入 RTSP 流 URL
#             '-c:v', 'libx264',  # 编码器
#             '-f', 'rtsp',  # 输出格式为 RTSP
#             '-loglevel', 'debug',  # 添加调试输出
#             rtsp_target_url  # RTSP 目标 URL
#         ]
#         logging.info(f"推送视频流命令: {' '.join(ffmpeg_command)}")
        
#         # 启动 FFmpeg 子进程
#         process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

#         # 打开输入流
#         #cap = cv2.VideoCapture(rtsp_url)
#         cap = cv2.VideoCapture(rtsp_url)
#         cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 设置较小的缓冲区
#         logging.warning("无法读取视频帧111111111111")
#         while True:
#             logging.warning("无法读取视频帧22222222222222")
#             ret, frame = cap.read()
#             if not ret:
#                 logging.warning("无法读取视频帧")
#                 break
#             logging.warning("无法读取视频帧33333333333333333")
#             # 在画面中间绘制红色矩形框
#             height, width, _ = frame.shape
#             cv2.rectangle(frame, (int((width - 100) / 2), int((height - 100) / 2)), 
#                               (int((width + 100) / 2), int((height + 100) / 2)), (0, 0, 255), -1)

#             # 将结果帧发送到 FFmpeg 进程的标准输入
#             # 将结果帧发送到 FFmpeg 进程的标准输入
#             logging.warning("无法读取视频帧444444444444")
#             try:
#                 process.stdin.write(frame.tobytes())
#             except IOError as e:
#                 logging.error(f"写入数据时发生错误: {e}")
#                 break

#             # 显示帧（可选）
#             cv2.imshow('RTSP Stream with Rectangle', frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         # 清理
#         cap.release()
#         cv2.destroyAllWindows()
#         process.stdin.close()

#         # 读取 FFmpeg 的 stderr 输出以查看错误信息
#         stderr_output = process.stderr.read()
#         if stderr_output:
#             logging.error(f"FFmpeg 错误信息: {stderr_output.decode()}")


#         process.wait()

#     except Exception as e:
#         logging.error(f"发生错误: {e}")

# # 使用示例
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.DEBUG)
#     rtsp_input_url = "rtsp://10.10.11.16:8556/live.sdp"  # 替换为输入 RTSP 流的 URL
#     rtsp_target_url = "rtsp://10.10.9.89:8557/live.sdp"  # 替换为目标 RTSP 流的 URL
#     stream_rtsp_with_rectangle(rtsp_input_url, rtsp_target_url)

# import cv2
# import subprocess
# import logging
# import numpy as np
# import os
# os.environ['DISPLAY'] = '10.10.11.16:0'

# def stream_rtsp_with_rectangle(rtsp_url, rtsp_target_url):
#     """
#     从 RTSP 流中获取视频，绘制矩形框，并推送到另一个 RTSP 地址
#     :param rtsp_url: 输入 RTSP 流的 URL
#     :param rtsp_target_url: 输出 RTSP 流的目标 URL
#     """
#     try:
#         # 打开输入流
#         cap = cv2.VideoCapture(rtsp_url)
#         if not cap.isOpened():
#             logging.error("无法打开输入 RTSP 流")
#             return

#         # 获取视频流的宽度和高度
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         fps = cap.get(cv2.CAP_PROP_FPS)

#         # 启动 FFmpeg 以推送视频流
#         ffmpeg_command = [
#             'ffmpeg',
#             '-y',  # 覆盖输出文件
#             '-f', 'rawvideo',  # 输入格式为原始视频
#             '-vcodec', 'rawvideo',
#             '-pix_fmt', 'bgr24',  # 像素格式
#             '-s', f'{width}x{height}',  # 设置分辨率
#             '-r', str(fps),  # 设置帧率
#             '-i', '-',  # 从标准输入读取数据
#             '-c:v', 'libx264',  # 使用 H.264 编码
#             '-f', 'rtsp',  # 输出格式为 RTSP
#             '-rtsp_transport', 'tcp',  # 使用 TCP 传输 RTSP
#             '-timeout', '5000000',  # 增加超时设置（5秒）
#             rtsp_target_url
#         ]
        
#         logging.info(f"推送视频流命令: {' '.join(ffmpeg_command)}")

#         # 启动 FFmpeg 子进程
#         process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE)

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 logging.warning("无法读取视频帧")
#                 break

#             # 在画面中间绘制红色矩形框
#             cv2.rectangle(frame, (int((width - 100) / 2), int((height - 100) / 2)), 
#                           (int((width + 100) / 2), int((height + 100) / 2)), (0, 0, 255), -1)

#             # 将结果帧发送到 FFmpeg 进程的标准输入
#             try:
#                 process.stdin.write(frame.tobytes())
#             except IOError as e:
#                 logging.error(f"写入数据时发生错误: {e}")
#                 break

#             # 显示帧（可选）
#             cv2.imshow('RTSP Stream with Rectangle', frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         # 清理
#         cap.release()
#         cv2.destroyAllWindows()
#         process.stdin.close()
#         process.wait()

#     except Exception as e:
#         logging.error(f"发生错误: {e}")

# # 使用示例
# if __name__ == "__main__":

#     logging.basicConfig(level=logging.DEBUG)
#     rtsp_input_url = "rtsp://10.10.11.16:8556/live.sdp"  # 替换为输入 RTSP 流的 URL
#     rtsp_target_url = "rtsp://10.10.9.89:8559/live.sdp"  # 替换为目标 RTSP 流的 URL
#     stream_rtsp_with_rectangle(rtsp_input_url, rtsp_target_url)




# import gi
# import numpy as np
# import cv2
# gi.require_version('Gst', '1.0')
# from gi.repository import Gst, GLib

# Gst.init(None)

# def start_rtsp_pipeline(rtsp_input_url, rtsp_output_url):
#     # 定义 GStreamer 管道，用来接收 RTSP 流并推送处理后的 RTSP 流
#     pipeline = Gst.parse_launch(f"""
#         rtspsrc location={rtsp_input_url} ! 
#         rtph264depay ! 
#         h264parse ! 
#         avdec_h264 ! 
#         videoconvert ! 
#         appsink name=sink
#     """)

#     # 获取 appsink 元素，接收视频帧
#     appsink = pipeline.get_by_name('sink')
#     appsink.set_property('emit-signals', True)
#     appsink.set_property('sync', False)

#     # 回调函数：处理每个视频帧
#     def on_new_sample(sink):
#         sample = sink.emit('pull-sample')
#         buf = sample.get_buffer()

#         # 获取帧数据
#         caps = sample.get_caps()
#         frame_format = caps.get_structure(0).get_value('format')
#         width = caps.get_structure(0).get_value('width')
#         height = caps.get_structure(0).get_value('height')
        
#         # 将视频帧转换为 numpy 数组
#         success, map_info = buf.map(Gst.MapFlags.READ)
#         if not success:
#             return Gst.FlowReturn.ERROR
        
#         frame = np.frombuffer(map_info.data, np.uint8).reshape(height, width, 3)
#         buf.unmap(map_info)

#         # 在中心绘制 100x100 的红色矩形框
#         center_x, center_y = width // 2, height // 2
#         top_left = (center_x - 50, center_y - 50)
#         bottom_right = (center_x + 50, center_y + 50)
#         cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), -1)

#         # 将处理后的帧传递到 appsrc 进行推流
#         push_frame_to_rtsp(frame, width, height, rtsp_output_url)

#         return Gst.FlowReturn.OK

#     # 将回调函数连接到 appsink
#     appsink.connect('new-sample', on_new_sample)

#     # 启动管道
#     pipeline.set_state(Gst.State.PLAYING)

#     # 保持主循环运行
#     loop = GLib.MainLoop()
#     loop.run()

# def push_frame_to_rtsp(frame, width, height, rtsp_output_url):
#     # 定义推流的 GStreamer 管道
#     pipeline_push = Gst.parse_launch(f"""
#         appsrc name=src ! 
#         videoconvert ! 
#         x264enc tune=zerolatency ! 
#         rtph264pay config-interval=1 pt=96 ! 
#         udpsink host={rtsp_output_url} port=8554
#     """)
    
#     # 获取 appsrc 元素
#     appsrc = pipeline_push.get_by_name('src')
    
#     # 设置 appsrc 的属性
#     appsrc.set_property('caps', Gst.Caps.from_string(f'video/x-raw,format=BGR,width={width},height={height},framerate=30/1'))
#     appsrc.set_property('block', True)

#     # 启动推流管道
#     pipeline_push.set_state(Gst.State.PLAYING)

#     # 将帧数据推送到 appsrc
#     buffer = Gst.Buffer.new_wrapped_bytes(frame.tobytes())
#     appsrc.emit('push-buffer', buffer)

# if __name__ == "__main__":
#     rtsp_input_url = "rtsp://10.10.11.16:8556/live.sdp"  # 从 PC 接收 RTSP 流
#     rtsp_output_url = "rtsp://10.10.9.89:8557/live.sdp"  # 推送回 PC 的 RTSP 目标地址
#     start_rtsp_pipeline(rtsp_input_url, rtsp_output_url)


# import gi
# import numpy as np
# import cv2
# gi.require_version('Gst', '1.0')
# from gi.repository import Gst, GLib

# Gst.init(None)

# # 定义全局变量存储推流管道
# pipeline_push = None

# def start_rtsp_pipeline(rtsp_input_url, rtsp_output_url):
#     global pipeline_push  # 声明使用全局变量
    
#     # 定义 GStreamer 管道，用来接收 RTSP 流并推送处理后的 RTSP 流
#     pipeline = Gst.parse_launch(f"""
#         rtspsrc location={rtsp_input_url} ! 
#         rtph264depay ! 
#         h264parse ! 
#         avdec_h264 ! 
#         videoconvert ! 
#         appsink name=sink
#     """)

#     # 获取 appsink 元素，接收视频帧
#     appsink = pipeline.get_by_name('sink')
#     appsink.set_property('emit-signals', True)
#     appsink.set_property('sync', False)

#     # 回调函数：处理每个视频帧
#     def on_new_sample(sink):
#         sample = sink.emit('pull-sample')
#         buf = sample.get_buffer()

#         # 获取帧数据
#         caps = sample.get_caps()
#         width = caps.get_structure(0).get_value('width')
#         height = caps.get_structure(0).get_value('height')
        
#         # 将视频帧转换为 numpy 数组
#         success, map_info = buf.map(Gst.MapFlags.READ)
#         if not success:
#             return Gst.FlowReturn.ERROR
        
#         # 调整数据形状
#         expected_size = height * width * 3
#         if map_info.size != expected_size:
#             print(f"Warning: Expected size {expected_size}, but got {map_info.size}")
#             buf.unmap(map_info)
#             return Gst.FlowReturn.ERROR
        
#         frame = np.frombuffer(map_info.data, np.uint8).reshape(height, width, 3)
#         buf.unmap(map_info)

#         # 在中心绘制 100x100 的红色矩形框
#         center_x, center_y = width // 2, height // 2
#         top_left = (center_x - 50, center_y - 50)
#         bottom_right = (center_x + 50, center_y + 50)
#         cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), -1)

#         # 将处理后的帧传递到 appsrc 进行推流
#         push_frame_to_rtsp(frame, width, height, rtsp_output_url)

#         return Gst.FlowReturn.OK

#     # 将回调函数连接到 appsink
#     appsink.connect('new-sample', on_new_sample)

#     # 启动管道
#     pipeline.set_state(Gst.State.PLAYING)

#     # 保持主循环运行
#     loop = GLib.MainLoop()
#     loop.run()

# def push_frame_to_rtsp(frame, width, height, rtsp_output_url):
#     global pipeline_push  # 使用全局变量

#     # 如果管道还未创建，则创建
#     if pipeline_push is None:
#         # 定义推流的 GStreamer 管道
#         pipeline_push = Gst.parse_launch(f"""
#             appsrc name=src ! 
#             videoconvert ! 
#             x264enc tune=zerolatency ! 
#             rtph264pay config-interval=1 pt=96 ! 
#             udpsink host={rtsp_output_url.split(':')[1]} port=8557
#         """)
        
#         # 获取 appsrc 元素
#         appsrc = pipeline_push.get_by_name('src')
        
#         # 设置 appsrc 的属性
#         appsrc.set_property('caps', Gst.Caps.from_string(f'video/x-raw,format=BGR,width={width},height={height},framerate=30/1'))
#         appsrc.set_property('block', True)

#         # 启动推流管道
#         pipeline_push.set_state(Gst.State.PLAYING)

#     # 将帧数据推送到 appsrc
#     buffer = Gst.Buffer.new_wrapped_bytes(frame.tobytes())
#     appsrc.emit('push-buffer', buffer)

# if __name__ == "__main__":
#     rtsp_input_url = "rtsp://10.10.11.16:8556/live.sdp"  # 从 PC 接收 RTSP 流
#     rtsp_output_url = "rtsp://10.10.9.89:8557/live.sdp"  # 推送回 PC 的 RTSP 目标地址
#     start_rtsp_pipeline(rtsp_input_url, rtsp_output_url)


# import gi
# import numpy as np
# import cv2
# gi.require_version('Gst', '1.0')
# from gi.repository import Gst, GLib

# Gst.init(None)

# # 定义全局变量存储推流管道
# pipeline_push = None

# def start_rtsp_pipeline(rtsp_input_url, rtsp_output_url):
#     global pipeline_push  # 声明使用全局变量
    
#     # 定义 GStreamer 管道，用来接收 RTSP 流并推送处理后的 RTSP 流
#     pipeline = Gst.parse_launch(f"""
#         rtspsrc location={rtsp_input_url} ! 
#         rtph264depay ! 
#         h264parse ! 
#         avdec_h264 ! 
#         videoconvert ! 
#         appsink name=sink
#     """)

#     # 获取 appsink 元素，接收视频帧
#     appsink = pipeline.get_by_name('sink')
#     appsink.set_property('emit-signals', True)
#     appsink.set_property('sync', False)

#     # 回调函数：处理每个视频帧
#     def on_new_sample(sink):
#         sample = sink.emit('pull-sample')
#         buf = sample.get_buffer()

#         # 获取帧数据
#         caps = sample.get_caps()
#         width = caps.get_structure(0).get_value('width')
#         height = caps.get_structure(0).get_value('height')
        
#         # 将视频帧转换为 numpy 数组
#         success, map_info = buf.map(Gst.MapFlags.READ)
#         if not success:
#             return Gst.FlowReturn.ERROR
        
#         # 调整数据形状
#         expected_size = height * width * 3
#         if map_info.size != expected_size:
#             print(f"Warning: Expected size {expected_size}, but got {map_info.size}")
#             buf.unmap(map_info)
#             return Gst.FlowReturn.ERROR
        
#         frame = np.frombuffer(map_info.data, np.uint8).reshape(height, width, 3)
#         buf.unmap(map_info)

#         # 在中心绘制 100x100 的红色矩形框
#         center_x, center_y = width // 2, height // 2
#         top_left = (center_x - 50, center_y - 50)
#         bottom_right = (center_x + 50, center_y + 50)
#         cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), -1)

#         # 将处理后的帧传递到 appsrc 进行推流
#         push_frame_to_rtsp(frame, width, height, rtsp_output_url)

#         return Gst.FlowReturn.OK

#     # 将回调函数连接到 appsink
#     appsink.connect('new-sample', on_new_sample)

#     # 启动管道
#     pipeline.set_state(Gst.State.PLAYING)

#     # 保持主循环运行
#     loop = GLib.MainLoop()
#     loop.run()

# def push_frame_to_rtsp(frame, width, height, rtsp_output_url):
#     global pipeline_push  # 使用全局变量

#     # 如果管道还未创建，则创建
#     if pipeline_push is None:
#         # 定义推流的 GStreamer 管道
#         pipeline_push = Gst.parse_launch(f"""
#             appsrc name=src ! 
#             videoconvert ! 
#             x264enc tune=zerolatency ! 
#             rtph264pay config-interval=1 pt=96 ! 
#             udpsink host={rtsp_output_url.split(':')[1]} port=8557
#         """)
        
#         # 获取 appsrc 元素
#         appsrc = pipeline_push.get_by_name('src')
        
#         # 设置 appsrc 的属性
#         appsrc.set_property('caps', Gst.Caps.from_string(f'video/x-raw,format=BGR,width={width},height={height},framerate=30/1'))
#         appsrc.set_property('block', True)

#         # 启动推流管道
#         pipeline_push.set_state(Gst.State.PLAYING)

#     # 将帧数据推送到 appsrc
#     buffer = Gst.Buffer.new_wrapped_bytes(frame.tobytes())
#     appsrc.emit('push-buffer', buffer)

# if __name__ == "__main__":
#     rtsp_input_url = "rtsp://10.10.11.16:8556/live.sdp"  # 从 PC 接收 RTSP 流
#     rtsp_output_url = "rtsp://10.10.9.89:8557/live.sdp"  # 推送回 PC 的 RTSP 目标地址
#     start_rtsp_pipeline(rtsp_input_url, rtsp_output_url)


import gi
import numpy as np
import cv2
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

Gst.init(None)

# 定义全局变量存储推流管道
pipeline_push = None

def start_rtsp_pipeline(rtsp_input_url, rtsp_output_url):
    global pipeline_push  # 声明使用全局变量
    
    # 定义 GStreamer 管道，用来接收 RTSP 流并推送处理后的 RTSP 流
    pipeline = Gst.parse_launch(f"""
        rtspsrc location={rtsp_input_url} ! 
        rtph264depay ! 
        h264parse ! 
        avdec_h264 ! 
        videoconvert ! 
        appsink name=sink
    """)

    # 获取 appsink 元素，接收视频帧
    appsink = pipeline.get_by_name('sink')
    appsink.set_property('emit-signals', True)
    appsink.set_property('sync', False)

    # 回调函数：处理每个视频帧
    def on_new_sample(sink):
        #print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxx1 New sample received .")
        sample = sink.emit('pull-sample')
        buf = sample.get_buffer()

        # 获取帧数据
        caps = sample.get_caps()
        width = caps.get_structure(0).get_value('width')
        height = caps.get_structure(0).get_value('height')
        
        # 将视频帧转换为 numpy 数组
        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            print("Failed to map buffer.")
            return Gst.FlowReturn.ERROR
        
        # 调整数据形状
        expected_size = height * width * 3
        if map_info.size != expected_size:
            #print(f"Warning: Expected size {expected_size}, but got {map_info.size}")
            buf.unmap(map_info)
            return Gst.FlowReturn.ERROR
        
        frame = np.frombuffer(map_info.data, np.uint8).reshape(height, width, 3)
        buf.unmap(map_info)

        # 在中心绘制 100x100 的红色矩形框
        center_x, center_y = width // 2, height // 2
        top_left = (center_x - 50, center_y - 50)
        bottom_right = (center_x + 50, center_y + 50)
        cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), -1)

        # 将处理后的帧传递到 appsrc 进行推流
        push_frame_to_rtsp(frame, width, height, rtsp_output_url)

        return Gst.FlowReturn.OK

    # 将回调函数连接到 appsink
    appsink.connect('new-sample', on_new_sample)

    # 启动管道
    pipeline.set_state(Gst.State.PLAYING)

# 检查管道状态
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, pipeline)

    print("GStreamer pipeline started.")

    # 保持主循环运行
    loop = GLib.MainLoop()
    loop.run()

def bus_call(bus, message, pipeline):
    if message.type == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"Error: {err}, {debug}")
    elif message.type == Gst.MessageType.EOS:
        print("End-of-stream reached.")
        pipeline.set_state(Gst.State.NULL)
        GLib.MainLoop.quit()
    elif message.type == Gst.MessageType.STATE_CHANGED:
        old_state, new_state, pending_state = message.parse_state_changed()
        print(f"Pipeline state changed from {old_state} to {new_state}")
    return True


def push_frame_to_rtsp(frame, width, height, rtsp_output_url):
    global pipeline_push  # 使用全局变量
    print("aaaaaaaaaaaaa.",pipeline_push)
    # 如果管道还未创建，则创建
    if pipeline_push is None:
        # 提取 IP 和端口
        host, port = rtsp_output_url.split(":")[1], rtsp_output_url.split(":")[2].split("/")[0]
        
        # 定义推流的 GStreamer 管道
        pipeline_push = Gst.parse_launch(f"""
            appsrc name=src ! 
            videoconvert ! 
            x264enc tune=zerolatency ! 
            rtph264pay config-interval=1 pt=96 ! 
            udpsink host={rtsp_output_url.split(':')[1]} port=8557
        """)
        
        # 获取 appsrc 元素
        appsrc = pipeline_push.get_by_name('src')
        
        # 设置 appsrc 的属性
        appsrc.set_property('caps', Gst.Caps.from_string(f'video/x-raw,format=BGR,width={width},height={height},framerate=30/1'))
        appsrc.set_property('block', True)
       
        # 启动推流管道
        print("yyyyyyyyyyyyyyyyyyyy2   Push pipeline state changed to PLAYING.")
        pipeline_push.set_state(Gst.State.PLAYING)
        print("xxxxxxxxxxxxxxx2    Push pipeline state changed to PLAYING.")

    # 将帧数据推送到 appsrc
    buffer = Gst.Buffer.new_wrapped_bytes(frame.tobytes())
    appsrc.emit('push-buffer', buffer)

if __name__ == "__main__":
    rtsp_input_url = "rtsp://10.10.11.16:8556/live.sdp"  # 从 PC 接收 RTSP 流
    rtsp_output_url = "rtsp://10.10.9.89:8557/live.sdp"  # 推送回 PC 的 RTSP 目标地址
    start_rtsp_pipeline(rtsp_input_url, rtsp_output_url)
