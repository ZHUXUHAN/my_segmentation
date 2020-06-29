from PIL import Image
import os

root_dir = '/train/trainset/1/Semantic/data/img'
save_dir = '/train/trainset/1/Semantic/data/clear_img'

paths = os.listdir(root_dir)
for path in paths:
    print(path)
    i = 1
    j = 1
    img = Image.open(os.path.join(root_dir, path))#读取系统的内照片
    width = img.size[0]#长度
    height = img.size[1]#宽度
    for i in range(0,width):#遍历所有长度的点
        for j in range(0,height):#遍历所有宽度的点
            data = (img.getpixel((i,j)))#打印该图片的所有点
            if data[0]==0 and data[0]==0 and data[0]==0:
                img.putpixel((i, j), (255, 0 , 0))  # 则这些像素点的颜色改成绿色
    #         if ((230<data[0]<250 and 170<data[1]<185 and 170<data[2]<190)or(180<data[0]<210 and 110<data[1]<130 and 120<data[2]<140)):#RGBA的r值大于170，并且g值大于170,并且b值大于170
    #             #判断条件就是一个像素范围范围
    #             # img.putpixel((i,j),(0,255,0))#则这些像素点的颜色改成绿色
    #             img.putpixel((i, j), (0, 128, 255))  # 则这些像素点的颜色改成蓝色

    img = img.convert("RGB")#把图片强制转成RGB
    img.save(os.path.join(save_dir, path))#保存修改像素点后的图片
