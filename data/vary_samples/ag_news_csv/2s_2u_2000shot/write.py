import os



stat_dict={}

write_res=[]

with open("./train.csv","r") as  f:
    for line in f.readlines():
        arr = line.split(',', 1)
        class_label = int(eval(arr[0]))
        if class_label not in stat_dict:
            stat_dict[class_label] = 0
        if stat_dict[class_label]>=2000: 
            continue
        else :
            stat_dict[class_label] += 1
            write_res.append(line)

with open("./test_mine.csv","w") as f:
    for line in write_res:
        f.write(line)
        
 