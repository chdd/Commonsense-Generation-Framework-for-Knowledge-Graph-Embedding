def split_file(file_path, output_first, output_last, num_lines=35851):
    with open(file_path, 'r',encoding='UTF-8') as file:
        lines = file.readlines()

    # 前三万行
    with open(output_first, 'w',encoding='UTF-8') as first_file:
        first_file.writelines(lines[:num_lines])

    # 后三万行
    with open(output_last, 'w',encoding='UTF-8') as last_file:
        last_file.writelines(lines[num_lines:])

# 替换下面的路径和文件名
input_file = 'D:\Experiment\work\cge-hake\data_concept\dbpedia\\valid.txt'
output_first_file = 'first_30000_lines.txt'
output_last_file = 'last_30000_lines.txt'
split_file(input_file, output_first_file, output_last_file)