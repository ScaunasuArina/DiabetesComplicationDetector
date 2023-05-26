from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score
import time

# ==============================================
#               SVM Model
#             with Classes
# ==============================================
class SVMModel:
    def __int__(self):
        pass

    def fit_the_model(self, X_train, y_train):
        # svm_model = svm.SVC(kernel='rbf')
        self.svm_model = svm.SVC(kernel='linear')
        print("\nFitting the model...")

        start_time = time.time()
        self.svm_model.fit(X_train, y_train)
        stop_time = time.time()

        print(f"Start time: {start_time}\n")
        print(f"Stop time: {stop_time}\n")
        print(f"Training duration: {stop_time - start_time} seconds.")

    def predict_value_and_return_accuracy(self, X_test, y_test):
        model_predict = self.svm_model.predict(X_test)

        print(f"CONFUSION MATRIX: {confusion_matrix(y_test, model_predict)}\n")
        print(f"Accuracy is {round(accuracy_score(y_test, model_predict) * 100, 2)}%\n")

    def predict_value(self, X_test):
        model_predict = self.svm_model.predict(X_test)
        return model_predict
