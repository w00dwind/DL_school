from sklearn.metrics import classification_report
import numpy as np
import pickle

def make_report(true, preds, total_report, label_encoder):
    #     print(total_report)
    true_names = label_encoder.inverse_transform(true)
    preds_names = label_encoder.inverse_transform(preds)

    fresh_report = classification_report(true_names, preds_names, output_dict=True, zero_division=1)
    if len(total_report) == 0:
        total_report = fresh_report
        return total_report
    for person, metrics in fresh_report.items():
        #         print(fresh_report[person])
        if person not in total_report.keys():
            total_report[person] = fresh_report[person]

    for person, metrics in fresh_report.items():
        if type(fresh_report[person]) != dict:
            total_report[person] = total_report[person] + fresh_report[person] / 2
            continue
        for metric, value in metrics.items():
            total_report[person][metric] = (total_report[person][metric] + fresh_report[person][metric]) / 2


    #     total_report.update(report)
    #     print(total_report)
    return total_report

np.random.seed(2023)

true = np.random.randint(42, size=8)
preds = np.random.randint(42, size=8)

le = pickle.load(open('/Users/ac1d/PycharmProjects/DL_school/20_Simpsons/label_encoder.pkl', 'rb'))

total_report = {}

make_report(true, preds, total_report, le)
# print(classification_report(true, preds))
# classification_report(true, preds)