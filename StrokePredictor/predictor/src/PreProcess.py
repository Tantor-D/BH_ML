import csv


def preprocess():
    # 这个函数负责将原始的train.csv文件转为trainDataset.csv文件
    # 会将其中的各种字符串变量转为数组
    # 此次预处理，会将N/A转变为0
    trans_dict = {
        'Female': 0,
        'Male': 1,
        'Other': 3,

        'No': 0,
        'Yes': 1,

        'Self-employed': 0,
        'Private': 1,
        'Govt_job': 2,
        'children': 3,
        'Never_worked': 4,

        'Rural': 0,
        'Urban': 1,

        'never smoked': -1,
        'Unknown': 0,
        'smokes': 1,
        'formerly smoked': 2
    }

    with open('../data/train.csv', 'r') as f:
        reader = csv.DictReader(f)
        with open('../data/trainDataSet.csv', 'w', newline='') as g:
            writer = csv.writer(g)
            for (row) in reader:
                thisRow = []
                thisRow.append(trans_dict[row['gender']])
                thisRow.append(row['age'])
                thisRow.append(row['hypertension'])
                thisRow.append(row['heart_disease'])
                thisRow.append(trans_dict[row['ever_married']])
                thisRow.append(trans_dict[row['work_type']])
                thisRow.append(trans_dict[row['Residence_type']])
                thisRow.append(row['avg_glucose_level'])
                if (row['bmi'] == 'N/A'):
                    thisRow.append(0)
                else:
                    thisRow.append(row['bmi'])
                thisRow.append(trans_dict[row['smoking_status']])
                thisRow.append(row['stroke'])
                writer.writerow(thisRow)

    with open('../data/test.csv', 'r') as f:
        reader = csv.DictReader(f)
        with open('../data/testDataSet.csv', 'w', newline='') as g:
            writer = csv.writer(g)
            for (row) in reader:
                thisRow = []
                thisRow.append(trans_dict[row['gender']])
                thisRow.append(row['age'])
                thisRow.append(row['hypertension'])
                thisRow.append(row['heart_disease'])
                thisRow.append(trans_dict[row['ever_married']])
                thisRow.append(trans_dict[row['work_type']])
                thisRow.append(trans_dict[row['Residence_type']])
                thisRow.append(row['avg_glucose_level'])
                if (row['bmi'] == 'N/A'):
                    thisRow.append(0)
                else:
                    thisRow.append(row['bmi'])
                thisRow.append(trans_dict[row['smoking_status']])
                # thisRow.append(row['stroke'])
                writer.writerow(thisRow)


if __name__ == '__main__':
    preprocess()
