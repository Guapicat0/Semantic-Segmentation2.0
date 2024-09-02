import os
class BatchRename():
    # 定义函数执行图片的路径
    def __init__(self):
        self.path = 'pile/JPEGImages'

    # 定义函数实现重命名操作
    def rename(self):
        filelist = os.listdir(self.path)
        total_num = len(filelist)
        i = 1
        for item in filelist:
            if item.endswith('.jpg'):
                src = os.path.join(os.path.abspath(self.path), item)
                dst = os.path.join(os.path.abspath(self.path), str(i) + '.jpg')
                try:
                    os.rename(src, dst)
                    print
                    'converting %s to %s ...' % (src, dst)
                    i = i + 1
                except:
                    continue
        print('total %d to rename & converted %d jpgs' % (total_num, i))

    # 主函数调用


if __name__ == '__main__':
    demo = BatchRename()
    demo.rename()

