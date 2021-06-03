class Transformer():

    def __init__(self, X_transformer=None, y_transformer=None):

        self.X_transformer = X_transformer if X_transformer != None else self.empty_transformer
        self.y_transformer = y_transformer if y_transformer != None else self.empty_transformer



    def empty_transformer(self, *kwargs):
        return kwargs

    def transform_X(self, X_tr, X_ts):
        return self.X_transformer(X_tr, X_ts)

    def transform_y(self, y_tr, y_ts):
        return self.y_transformer(y_tr, y_ts)

    def get_transform_X(self):
        return self.transform_X

    def get_transform_y(self):
        return self.self.transform_y




