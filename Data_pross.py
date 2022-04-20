import pandas as pd

out = "E:\\学习\\科研论文\\DLP\\Dataset\\opsahl-southernwomen\\out.opsahl-southernwomen"
DataSet = []
with open(out) as file_obj:
    content = file_obj.readlines()
    for i in range(2, len(content)):
        # print(content[i].split())
        numbers = [int(x) for x in content[i].split()]
        DataSet.append(numbers)
print(DataSet)
# name = ['src', 'dst', 'weight', 'timestamp']
name = ['src', 'dst']
data = pd.DataFrame(columns=name, data=DataSet)
print(data)
data.to_csv("E:\\学习\\科研论文\\DLP\\Dataset\\Data\\southernwomen.csv")

# ##该模块适合edit-cowikiquote和unicodelang
# # out = "E:\\学习\\科研论文\\DLP\\Dataset\\edit-cowikiquote\\out.edit-cowikiquote"
# out = "E:\\学习\\科研论文\\DLP\\Dataset\\unicodelang\\out.unicodelang"
# DataSet = []
# with open(out) as file_obj:
#     content = file_obj.readlines()
#     for i in range(1, len(content)):
#         # print(content[i].split())
#         numbers = []
#         for x in content[i].split()[:2]:
#             numbers.append(int(x))
#         # for x in content[i].split()[2:3]:
#         #     numbers.append(float(x))
#         DataSet.append(numbers)
# print(DataSet)
# # name = ['src', 'dst', 'weight', 'timestamp']
# name = ['src', 'dst']
# data = pd.DataFrame(columns=name, data=DataSet)
# data.to_csv("E:\\学习\\科研论文\\DLP\\Dataset\\Data\\unicodelang.csv")
