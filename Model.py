from sklearn.model_selection import train_test_split
import numpy as np

class Sequence:
    def __init__(self, layers) -> None:
        self.layers = layers
        self.loss = ''

        # save just weight(exclude bias)
        self.debugCache = {
            'approx': np.array([]),
            'backprop': np.array([]),
        }

    def __forward(self, X, debugParams = None):
        input = X
        for i, l in enumerate(self.layers):
            if debugParams is not None and i == debugParams['index']:
                output = l.forwardForGradApprox(input, debugParams['plus'])
            else:
                output = l.forward(input)
            input = output

        return output

    def __backward(self, Y, n_sampels):
        l = len(self.layers)
        W, dZ = self.layers[l - 1].backwardOutputLayer(Y, n_sampels)
        for i in reversed(range(l - 1)):
            W, dZ = self.layers[i].backwardHiddenLayer(W, dZ, n_sampels)
            # self.debugCache['backprop'] = np.append(dW, self.debugCache['backprop'])

    def __loss(self, output, Y, n_samples):
        if self.loss == 'binaryCrossEntropy':
            print(output)
            return -1. * ((Y * np.log(output)).sum() / n_samples)
        else:
            print('Model.Sequence: unknown loss')

    def __accuracy(self, output, Y, n_sampeles):
        return (output.argmax(0) == Y.argmax(0)).sum() / n_sampeles

    def compile(self, loss):
        self.loss = loss

    def fit(self, X, Y, learning_rate = 0.1, epoch = 10):
        # X:  (n_samples, n_features)
        # Y:  (n_samples, 1)

        for l in self.layers:
            l.set_params(learning_rate)

        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        # XNorm = (X - self.mean) / self.std

        for i in range(epoch):
            train_x, valid_x, train_y, valid_y = train_test_split(X, Y, train_size=0.8, shuffle=True)
            n_trains = train_y.shape[0]
            n_valids = valid_y.shape[0]

            # for ease of calculation
            train_x, train_y = train_x.T, train_y.T
            valid_x, valid_y = valid_x.T, valid_y.T

            # debug
            # for j, l in enumerate(self.layers):
            #     outputPlus = self.__forward(train_x, {'index': j, 'plus': True })
            #     outputMinus = self.__forward(train_x, {'index': j, 'plus': False })
            #     outputLossPlus = self.__loss(outputPlus, train_y, n_trains)
            #     outputLossMinus = self.__loss(outputMinus, train_y, n_trains)
            #     gradApprox = (-outputLossMinus + outputLossPlus) / 0.02
                # self.debugCache['approx'] = np.append(self.debugCache['approx'], gradApprox)

            output = self.__forward(train_x)
            self.__backward(train_y, n_trains)

            # training loss and accuracy
            train_output = self.__forward(train_x)
            train_loss = self.__loss(train_output, train_y, n_trains)
            train_accuracy = self.__accuracy(train_output, train_y, n_trains)

            # validation loss and accuracy
            valid_output = self.__forward(valid_x)
            valid_loss = self.__loss(valid_output, valid_y, n_valids)
            valid_accuracy = self.__accuracy(valid_output, valid_y, n_valids)

            print(
                f'epoch {i + 1}/{epoch}: - loss: {train_loss:.4} - acc: {train_accuracy:.4} - val_loss: {valid_loss:.4} - val_acc: {valid_accuracy:.4}')
