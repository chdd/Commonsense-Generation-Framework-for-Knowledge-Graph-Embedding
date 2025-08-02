def tsv_to_txt(input_file, output_file):
    try:
        # 读取TSV文件
        with open(input_file, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()

        # 写入TXT文件
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for line in lines:
                outfile.write(line)

        print(f"Converted {input_file} to {output_file} successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")


# 文件名
input_file = r'D:\Experiment\work\cge-hake\data_concept\umls\test.tsv'
output_file = r'D:\Experiment\work\cge-hake\data_concept\umls\test.txt'

# 转换TSV到TXT
tsv_to_txt(input_file, output_file)