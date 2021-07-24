# Custom importing
from parameters import *
from painting_retrieval import PaintingRetrieval
from read_write import find_image


def check_image_retrieval(folder=PATH_TEST_DATASET):
    true_positive, false_negative, true_negative, false_positive = 0, 0, 0, 0
    path_paintings = find_image(folder)

    print(f"Found {len(path_paintings)} test paintings")
    print("Start Retrieval ...")
    print('*' * 25)
    painting_retrieval = PaintingRetrieval()
    path_paintings.sort()
    print('*' * 25)
    for name_image in path_paintings:
        list_retrieval = painting_retrieval.match_painting(name_image)
        if len(list_retrieval) > 0:
            best_match, similarity = list_retrieval[0][0], round(list_retrieval[0][1], 3)
            painting_test = (os.path.splitext(os.path.basename(name_image))[0]).split('_')[0]
            if similarity <= 0.065:
                predicted_paintings = 'NonMatch'
            else:
                predicted_paintings = best_match.split('.')[0]
            if painting_test != 'NonMatch' and painting_test == predicted_paintings:
                true_positive += 1
            elif painting_test != 'NonMatch' and painting_test != predicted_paintings:
                false_positive += 1
            elif painting_test == 'NonMatch' and painting_test == predicted_paintings:
                true_negative += 1
            elif painting_test == 'NonMatch' and painting_test != predicted_paintings:
                false_negative += 1
            else:
                print("ERROR!")
            print(f"Test: {painting_test} - Predicted: {predicted_paintings} with similarity: {similarity}")


    print(true_positive, false_positive, true_negative, false_negative)
    accuracy = round((true_positive + true_negative) / len(path_paintings), 3)
    recall = round(true_positive / (true_positive + false_negative), 3)
    precision = round(true_positive / (true_positive + false_positive), 3)
    tpr = round(true_positive / (true_positive + true_negative), 3)
    tnr = round(false_positive / (false_positive + true_negative), 3)
    f1_score = round(2 * (precision * recall) / (precision + recall), 3)
    print(f"Accuracy: {accuracy} - Recall: {recall} - Precision: {precision} - F1 Score: {f1_score}")
    print(f"True Accettable Rate : {tpr} - False Accettable Rate {tnr}")


try:
    check_image_retrieval()
except KeyboardInterrupt:
    print("Exit!")
