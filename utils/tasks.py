def get_dataset_list(dataset, mode, overlap=True):
    
    all_dataset = open(f"datasets/data/{dataset}/{mode}_cls.txt", "r").read().splitlines()
    
    target_cls = list(range(21))
    
    if 0 in target_cls:
        target_cls.remove(0)
    
    dataset_list = []
    
    if overlap:
        fil = lambda c: any(x in target_cls for x in classes)
    else:
        target_cls_old = list(range(1, target_cls[0]))
        target_cls_cum = target_cls + target_cls_old + [0, 255]

        fil = lambda c: any(x in target_cls for x in classes) and all(x in target_cls_cum for x in c)
    
    for idx, classes in enumerate(all_dataset):
        str_split = classes.split(" ")

        img_name = str_split[0]
        classes = [int(s)+1 for s in str_split[1:]]

        if fil(classes):
            dataset_list.append(img_name)
            
    return dataset_list