# from collections import OrderedDict

# # d = OrderedDict()
# d = dict()
# d['b'] = 2
# d['a'] = 1
# d['d'] = 1
# d['c'] = 1
# print(d)

# newkeys = sorted(d.keys())
# print(newkeys)
# sorted_dict = {}
# for k in newkeys:
#     sorted_dict[k] = d[k]
# print(sorted_dict)


a = [(1,2), (3,4), (5,6)]
a_ = [i for i, j in a]
print(argmax(a_))