
from django.shortcuts import render
import json
import os
from django.conf import settings
import pandas as pd
from sklearn import model_selection, svm, naive_bayes
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

def task1_view(request):
    results_path = os.path.join(settings.MEDIA_ROOT, 'model_results.json')

    if os.path.exists(results_path):
        try:
            with open(results_path, 'r') as f:
                results = json.load(f)
            print("Loaded results from JSON file.")
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            return render(request, 'users/task1.html', {'error': 'Failed to load previous results.'})
    else:
        try:
            path1 = os.path.join(settings.MEDIA_ROOT, 'processed_data_vol2.csv')
            path2 = os.path.join(settings.MEDIA_ROOT, 'class.csv')
            dp = pd.read_csv(path1, encoding='cp1252')
            dc = pd.read_csv(path2, encoding='cp1252')
        except Exception as e:
            print(f"Error reading data: {e}")
            return render(request, 'users/task1.html', {'error': 'Failed to read data.'})

        try:
            Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(
                dp['text_final'], dc['class'], test_size=0.3)
        except Exception as e:
            print(f"Error splitting data: {e}")

        try:
            Encoder = LabelEncoder()
            Train_Y = Encoder.fit_transform(Train_Y)
            Test_Y = Encoder.transform(Test_Y)
        except Exception as e:
            print(f"Error encoding labels: {e}")

        try:
            Tfidf_vect = TfidfVectorizer()
            Tfidf_vect.fit(dp['text_final'])
            Train_X_Tfidf = Tfidf_vect.transform(Train_X)
            Test_X_Tfidf = Tfidf_vect.transform(Test_X)
        except Exception as e:
            print(f"Error vectorizing text data: {e}")

        try:
            SVM = svm.SVC(C=1.0, kernel='linear', gamma='auto')
            SVM.fit(Train_X_Tfidf, Train_Y)

            Naive = naive_bayes.MultinomialNB()
            Naive.fit(Train_X_Tfidf, Train_Y)

            pickle.dump(SVM, open(os.path.join(settings.MEDIA_ROOT, 'finalized_model_SVM.sav'), 'wb'))
            pickle.dump(Naive, open(os.path.join(settings.MEDIA_ROOT, 'finalized_model_NB.sav'), 'wb'))
        except Exception as e:
            print(f"Error training or saving models: {e}")

        predictions_SVM = SVM.predict(Test_X_Tfidf)
        predictions_NB = Naive.predict(Test_X_Tfidf)

        svm_accuracy = accuracy_score(Test_Y, predictions_SVM) * 100
        nb_accuracy = accuracy_score(Test_Y, predictions_NB) * 100

        def generate_conf_matrix(model, y_true, y_pred):
            mat = confusion_matrix(y_true, y_pred)
            labels = ['Hateful', 'Not Hateful']
            plt.figure(figsize=(6, 4))
            sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
                        xticklabels=labels, yticklabels=labels)
            plt.title(f"{model} Confusion Matrix")
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()
            filename = f"{model}_confusion_matrix.png"
            plt.savefig(os.path.join(settings.MEDIA_ROOT, filename))
            plt.close()
            return filename

        svm_matrix_filename = generate_conf_matrix("SVM", Test_Y, predictions_SVM)
        nb_matrix_filename = generate_conf_matrix("Naive_Bayes", Test_Y, predictions_NB)

        results = {
            'svm_accuracy': svm_accuracy,
            'nb_accuracy': nb_accuracy,
            'svm_conf_matrix': svm_matrix_filename,
            'nb_conf_matrix': nb_matrix_filename,
        }

        try:
            with open(results_path, 'w') as f:
                json.dump(results, f)
        except Exception as e:
            print(f"Error saving results: {e}")
    results['MEDIA_URL'] = settings.MEDIA_URL

    return render(request, 'users/task1.html', results)


from django.conf import settings

def confusion_matrix_view(request):
    results_path = os.path.join(settings.MEDIA_ROOT, 'model_results.json')
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
        svm_image = results.get('svm_conf_matrix')
        nb_image = results.get('nb_conf_matrix')

        return render(request, 'users/confusion_matrix.html', {
            'svm_image': svm_image,
            'nb_image': nb_image,
            'MEDIA_URL': settings.MEDIA_URL,
        })
    else:
        return render(request, 'users/confusion_matrix.html', {
            'error': 'No results available yet.',
            'MEDIA_URL': settings.MEDIA_URL
        })
