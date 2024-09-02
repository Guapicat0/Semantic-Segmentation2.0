pile(数据集名称)

  -before

      -1.json
      -2.json
      ....

  -ImageSets

    -Segmentation
       -test.txt
       -train.txt
       -trainval.txt
       -val.txt

  -JPEGImages

      -1.jpg
      -2.jpg
       ...

  -SegmentationClass

      -1.png  #before文件中的json标准文件，通过json_to_dataset.py文件将其转化为标签文件图
      -2.png
      ...