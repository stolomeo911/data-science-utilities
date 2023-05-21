from sklearn.metrics import classification_report


def get_classification_report(y, predictions):
    report = classification_report(y, predictions, output_dict=True)
    print(report)
    return report
