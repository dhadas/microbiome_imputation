class Transformer():

    def __init__(self, X_transformer, Y_transformer):
        self.X_transformer = X_transformer
        self.Y_transformer = Y_transformer

    def __init__(self, **kwargs):
        print(kwargs)
        if len(kwargs) < 2:
            self.X_transformer = self.empty_transformer
            self.Y_transformer = self.empty_transformer
        else:
            self.X_transformer = kwargs[0]
            self.Y_transformer = kwargs[1]

    def empty_transformer(self, *kwargs):
        return kwargs

    def transform_X(self, X_tr, X_ts):
        return self.X_transformer(X_tr, X_ts)

    def transform_Y(self, y_tr, y_ts):
        return self.y_transformer(y_tr, y_ts)

    def get_transform_X(self):
        return self.transform_X

    def get_transform_y(self):
        return self.self.transform_y

