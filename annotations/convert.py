# import scipy.io
# import json

# data = scipy.io.loadmat("MPII_annotations.mat", struct_as_record = False)

# print(type(data))
# # json.dumps(scipy.io.loadmat("testdouble_7.4_GLNX86.mat")["testdouble"].tolist())
import scipy.io
import numpy as np
import json

decoded1 = scipy.io.loadmat("MPII_annotations.mat", struct_as_record=False)["RELEASE"]

must_be_list_fields = ["annolist", "annorect", "point", "img_train", "single_person", "act", "video_list"]

def generate_dataset_obj(obj):
    if type(obj) == np.ndarray:
        dim = obj.shape[0]
        if dim == 1:
            ret = generate_dataset_obj(obj[0])
        else:
            ret = []
            for i in range(dim):
                ret.append(generate_dataset_obj(obj[i]))

    elif type(obj) == scipy.io.matlab.mio5_params.mat_struct:
        ret = {}
        for field_name in obj._fieldnames:
            field = generate_dataset_obj(obj.__dict__[field_name])
            if field_name in must_be_list_fields and type(field) != list:
                field = [field]
            ret[field_name] = field

    else:
        ret = obj

    return ret

def print_dataset_obj(obj, depth = 0, maxIterInArray = 20):
    prefix = "  "*depth
    if type(obj) == dict:
        for key in obj.keys():
            print("{}{}".format(prefix, key))
            print_dataset_obj(obj[key], depth + 1)
    elif type(obj) == list:
        for i, value in enumerate(obj):
            if i >= maxIterInArray:
                break
            print("{}{}".format(prefix, i))
            print_dataset_obj(value, depth + 1)
    # else:
    #     print("{}{}".format(prefix, obj))

def convert(o):
    if isinstance(o, numpy.int64): return int(o)  
    raise TypeError


# Convert to dict
dataset_obj = generate_dataset_obj(decoded1)

# import csv

# w = csv.writer(open("output.csv", "w"))
# for key, val in dataset_obj.items():
#     w.writerow([key, val])

#with open('data.txt', 'w') as outfile:
    

# Print it out
print_dataset_obj(dataset_obj)