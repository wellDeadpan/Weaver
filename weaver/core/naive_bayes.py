#加入保存/加载功能
#加入 predict_proba() 支持
#多种 NB 类型切换（Bernoulli, Multinomial）

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

class HeartFailureNB:
    def __init__(self):
        self.model = GaussianNB()

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        preds = self.predict(X)
        return accuracy_score(y, preds)
