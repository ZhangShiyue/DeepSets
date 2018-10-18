# https://pyntcloud.readthedocs.io/en/latest/
# https://github.com/ZhangShiyue/DeepSets/tree/master/PointClouds
from pyntcloud import PyntCloud
#from sources.show3d_balls import showpoints
import h5py
import os

data_dir = 'ModelNet40/'
classls = os.listdir(data_dir)


'''
my_point_cloud = PyntCloud.from_file("airplane_0627.off")
sample = my_point_cloud.get_sample('mesh_random',n=1000)

cmd=showpoints(sample.values)
'''

train_pc = []
train_label = []
test_pc = []
test_label = []

f = open('sources/label.txt', "w")
for i,name in enumerate(classls):
    f.write(name + '\n')        
    print(name)
    tmp_dir = data_dir + name + '/train/'
    filels = os.listdir(tmp_dir)
    for model in filels:
        
        #print(tmp_dir + model)  
        m = PyntCloud.from_file(tmp_dir + model)
        pc = m.get_sample('mesh_random',n=10000).values
        train_pc.append(pc)
        train_label.append(i)          
    
    tmp_dir = data_dir + name + '/test/'
    filels = os.listdir(tmp_dir)
    for model in filels:
        #print(tmp_dir + model)  
        m = PyntCloud.from_file(tmp_dir + model)
        pc = m.get_sample('mesh_random',n=10000).values
        test_pc.append(pc)
        test_label.append(i)
        
        
f.close()


print('training data: %d' % len(train_pc))
print('training pc points: %d' % len(train_pc[0]))
print('training label: %d' % len(train_label))
print('testing data: %d' % len(test_pc))
print('testing pc points: %d' % len(test_pc[0]))
print('testing label: %d' % len(test_label))

hf = h5py.File('sources/data.h5', 'w')
hf.create_dataset('tr_cloud', data=train_pc)
hf.create_dataset('tr_label', data=train_label)
hf.create_dataset('test_cloud', data=test_pc)
hf.create_dataset('test_label', data=test_label)
hf.close()