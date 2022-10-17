from flask import Flask, render_template, request, redirect
import numpy as np
from sklearn import datasets

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/iris/', methods=('GET', 'POST'))
def iris():
    if request.method == 'POST':

        INPUT_DIM = 4
        OUT_DIM = 3
        H_DIM = 10

        def relu(t):
            return np.maximum(t, 0)

        def softmax_batch(t):
            out = np.exp(t)
            return out / np.sum(out, axis=1, keepdims=True)

        def sparse_cross_entropy_batch(z, y):
            return -np.log(np.array([z[j, y[j]] for j in range(len(y))]))

        def to_full_batch(y, num_classes):
            y_full = np.zeros((len(y), num_classes))
            for j, yj in enumerate(y):
                y_full[j, yj] = 1
            return y_full

        def relu_deriv(t):
            return (t >= 0).astype(float)

        iris = datasets.load_iris()
        dataset = [(iris.data[i][None, ...], iris.target[i]) for i in range(len(iris.target))]


        W1 = np.random.rand(INPUT_DIM, H_DIM)
        b1 = np.random.rand(1, H_DIM)
        W2 = np.random.rand(H_DIM, OUT_DIM)
        b2 = np.random.rand(1, OUT_DIM)

        W1 = (W1 - 0.5) * 2 * np.sqrt(1 / INPUT_DIM)
        b1 = (b1 - 0.5) * 2 * np.sqrt(1 / INPUT_DIM)
        W2 = (W2 - 0.5) * 2 * np.sqrt(1 / H_DIM)
        b2 = (b2 - 0.5) * 2 * np.sqrt(1 / H_DIM)

        ALPHA = 0.0002
        NUM_EPOCHS = 400
        BATCH_SIZE = 50

        loss_arr = []

        for ep in range(NUM_EPOCHS):
            for i in range(len(dataset) // BATCH_SIZE):
                batch_x, batch_y = zip(*dataset[i * BATCH_SIZE: i * BATCH_SIZE + BATCH_SIZE])
                x = np.concatenate(batch_x, axis=0)
                y = np.array(batch_y)

                # Forward
                t1 = x @ W1 + b1
                h1 = relu(t1)
                t2 = h1 @ W2 + b2
                z = softmax_batch(t2)
                E = np.sum(sparse_cross_entropy_batch(z, y))

                # Backward
                y_full = to_full_batch(y, OUT_DIM)
                dE_dt2 = z - y_full
                dE_dW2 = h1.T @ dE_dt2
                dE_db2 = np.sum(dE_dt2, axis=0, keepdims=True)
                dE_dh1 = dE_dt2 @ W2.T
                dE_dt1 = dE_dh1 * relu_deriv(t1)
                dE_dW1 = x.T @ dE_dt1
                dE_db1 = np.sum(dE_dt1, axis=0, keepdims=True)

                # Update
                W1 = W1 - ALPHA * dE_dW1
                b1 = b1 - ALPHA * dE_db1
                W2 = W2 - ALPHA * dE_dW2
                b2 = b2 - ALPHA * dE_db2

                loss_arr.append(E)

        def predict(x):
            t1 = x @ W1 + b1
            h1 = relu(t1)
            t2 = h1 @ W2 + b2
            z = softmax_batch(t2)
            return z

        # def calc_accuracy():
        #     correct = 0
        #     for x, y in dataset:
        #         z = predict(x)
        #         y_pred = np.argmax(z)
        #         if y_pred == y:
        #             correct += 1
        #     acc = correct / len(dataset)
        #     return acc

        # accuracy = calc_accuracy()
        # print("Accuracy:", accuracy)

        a = []
        sepal_length = int(request.form['sepal_length'])
        sepal_width = int(request.form['sepal_width'])
        petal_length = int(request.form['petal_length'])
        petal_width = int(request.form['petal_width'])

        a.append(sepal_length)
        a.append(sepal_width)
        a.append(petal_length)
        a.append(petal_width)

        x = np.array(a)


        def softmax(t):
            out = np.exp(t)
            return out / np.sum(out)

        def predict_otvet(x):
            t1 = x @ W1 + b1
            h1 = relu(t1)
            t2 = h1 @ W2 + b2
            z = softmax(t2)
            return z

        probs = predict_otvet(x)
        pred_class = np.argmax(probs)
        class_names = ['Setosa', 'Versicolor', 'Verginika']
        answer = class_names[pred_class]
        return render_template('iris.html', answer=answer)

    return render_template('iris.html')


@app.route('/breast_cancer/', methods=('GET', 'POST'))
def breast_cancer():
    if request.method == 'POST':

        INPUT_DIM = 30
        OUT_DIM = 2
        H_DIM = 10

        def relu(t):
            return np.maximum(t, 0)

        def softmax_batch(t):
            out = np.exp(t)
            return out / np.sum(out, axis=1, keepdims=True)

        def sparse_cross_entropy_batch(z, y):
            return -np.log(np.array([z[j, y[j]] for j in range(len(y))]))

        def to_full_batch(y, num_classes):
            y_full = np.zeros((len(y), num_classes))
            for j, yj in enumerate(y):
                y_full[j, yj] = 1
            return y_full

        def relu_deriv(t):
            return (t >= 0).astype(float)

        breast_cancer = datasets.load_breast_cancer()
        dataset = [(breast_cancer.data[i][None, ...], breast_cancer.target[i]) for i in range(len(breast_cancer.target))]

        W1 = np.random.rand(INPUT_DIM, H_DIM)
        b1 = np.random.rand(1, H_DIM)
        W2 = np.random.rand(H_DIM, OUT_DIM)
        b2 = np.random.rand(1, OUT_DIM)

        W1 = (W1 - 0.5) * 2 * np.sqrt(1 / INPUT_DIM)
        b1 = (b1 - 0.5) * 2 * np.sqrt(1 / INPUT_DIM)
        W2 = (W2 - 0.5) * 2 * np.sqrt(1 / H_DIM)
        b2 = (b2 - 0.5) * 2 * np.sqrt(1 / H_DIM)

        ALPHA = 0.0002
        NUM_EPOCHS = 400
        BATCH_SIZE = 50

        loss_arr = []

        for ep in range(NUM_EPOCHS):
            for i in range(len(dataset) // BATCH_SIZE):
                batch_x, batch_y = zip(*dataset[i * BATCH_SIZE: i * BATCH_SIZE + BATCH_SIZE])
                x = np.concatenate(batch_x, axis=0)
                y = np.array(batch_y)

                # Forward
                t1 = x @ W1 + b1
                h1 = relu(t1)
                t2 = h1 @ W2 + b2
                z = softmax_batch(t2)
                E = np.sum(sparse_cross_entropy_batch(z, y))

                # Backward
                y_full = to_full_batch(y, OUT_DIM)
                dE_dt2 = z - y_full
                dE_dW2 = h1.T @ dE_dt2
                dE_db2 = np.sum(dE_dt2, axis=0, keepdims=True)
                dE_dh1 = dE_dt2 @ W2.T
                dE_dt1 = dE_dh1 * relu_deriv(t1)
                dE_dW1 = x.T @ dE_dt1
                dE_db1 = np.sum(dE_dt1, axis=0, keepdims=True)

                # Update
                W1 = W1 - ALPHA * dE_dW1
                b1 = b1 - ALPHA * dE_db1
                W2 = W2 - ALPHA * dE_dW2
                b2 = b2 - ALPHA * dE_db2

                loss_arr.append(E)

        def predict(x):
            t1 = x @ W1 + b1
            h1 = relu(t1)
            t2 = h1 @ W2 + b2
            z = softmax_batch(t2)
            return z

        def calc_accuracy():
            correct = 0
            for x, y in dataset:
                z = predict(x)
                y_pred = np.argmax(z)
                if y_pred == y:
                    correct += 1
            acc = correct / len(dataset)
            return acc

        accuracy = calc_accuracy()
        print("Accuracy:", accuracy)

        a = []
        radius = int(request.form['radius'])
        texture = int(request.form['texture'])
        perimeter = int(request.form['perimeter'])
        area = int(request.form['area'])
        smoothness = int(request.form['smoothness'])
        compactness = int(request.form['compactness'])
        concavity = int(request.form['concavity'])
        concave_points = int(request.form['concave_points'])
        symmetry = int(request.form['symmetry'])
        fractal_dimension = int(request.form['fractal_dimension'])


        a.append(radius)
        a.append(texture)
        a.append(perimeter)
        a.append(area)
        a.append(smoothness)
        a.append(compactness)
        a.append(concavity)
        a.append(concave_points)
        a.append(symmetry)
        a.append(fractal_dimension)

        a.append(radius)
        a.append(texture)
        a.append(perimeter)
        a.append(area)
        a.append(smoothness)
        a.append(compactness)
        a.append(concavity)
        a.append(concave_points)
        a.append(symmetry)
        a.append(fractal_dimension)

        a.append(radius)
        a.append(texture)
        a.append(perimeter)
        a.append(area)
        a.append(smoothness)
        a.append(compactness)
        a.append(concavity)
        a.append(concave_points)
        a.append(symmetry)
        a.append(fractal_dimension)

        x = np.array(a)

        def softmax(t):
            out = np.exp(t)
            return out / np.sum(out)

        def predict_otvet(x):
            t1 = x @ W1 + b1
            h1 = relu(t1)
            t2 = h1 @ W2 + b2
            z = softmax(t2)
            return z

        probs = predict_otvet(x)
        pred_class = np.argmax(probs)
        class_names = ['WDBC-Malignant', 'WDBC-Benign']
        answer = class_names[pred_class]
        return render_template('breast_cancer.html', answer=answer)
    return render_template('breast_cancer.html')


@app.route('/wine/', methods=('GET', 'POST'))
def wine():
    if request.method == 'POST':
        INPUT_DIM = 13
        OUT_DIM = 3
        H_DIM = 10

        def relu(t):
            return np.maximum(t, 0)

        def softmax_batch(t):
            out = np.exp(t)
            return out / np.sum(out, axis=1, keepdims=True)

        def sparse_cross_entropy_batch(z, y):
            return -np.log(np.array([z[j, y[j]] for j in range(len(y))]))

        def to_full_batch(y, num_classes):
            y_full = np.zeros((len(y), num_classes))
            for j, yj in enumerate(y):
                y_full[j, yj] = 1
            return y_full

        def relu_deriv(t):
            return (t >= 0).astype(float)

        wine = datasets.load_wine()
        dataset = [(wine.data[i][None, ...], wine.target[i]) for i in range(len(wine .target))]

        W1 = np.random.rand(INPUT_DIM, H_DIM)
        b1 = np.random.rand(1, H_DIM)
        W2 = np.random.rand(H_DIM, OUT_DIM)
        b2 = np.random.rand(1, OUT_DIM)

        W1 = (W1 - 0.5) * 2 * np.sqrt(1 / INPUT_DIM)
        b1 = (b1 - 0.5) * 2 * np.sqrt(1 / INPUT_DIM)
        W2 = (W2 - 0.5) * 2 * np.sqrt(1 / H_DIM)
        b2 = (b2 - 0.5) * 2 * np.sqrt(1 / H_DIM)

        ALPHA = 0.00002
        NUM_EPOCHS = 1500
        BATCH_SIZE = 25

        loss_arr = []

        for ep in range(NUM_EPOCHS):
            for i in range(len(dataset) // BATCH_SIZE):
                batch_x, batch_y = zip(*dataset[i * BATCH_SIZE: i * BATCH_SIZE + BATCH_SIZE])
                x = np.concatenate(batch_x, axis=0)
                y = np.array(batch_y)

                # Forward
                t1 = x @ W1 + b1
                h1 = relu(t1)
                t2 = h1 @ W2 + b2
                z = softmax_batch(t2)
                E = np.sum(sparse_cross_entropy_batch(z, y))

                # Backward
                y_full = to_full_batch(y, OUT_DIM)
                dE_dt2 = z - y_full
                dE_dW2 = h1.T @ dE_dt2
                dE_db2 = np.sum(dE_dt2, axis=0, keepdims=True)
                dE_dh1 = dE_dt2 @ W2.T
                dE_dt1 = dE_dh1 * relu_deriv(t1)
                dE_dW1 = x.T @ dE_dt1
                dE_db1 = np.sum(dE_dt1, axis=0, keepdims=True)

                # Update
                W1 = W1 - ALPHA * dE_dW1
                b1 = b1 - ALPHA * dE_db1
                W2 = W2 - ALPHA * dE_dW2
                b2 = b2 - ALPHA * dE_db2

                loss_arr.append(E)

        def predict(x):
            t1 = x @ W1 + b1
            h1 = relu(t1)
            t2 = h1 @ W2 + b2
            z = softmax_batch(t2)
            return z

        def calc_accuracy():
            correct = 0
            for x, y in dataset:
                z = predict(x)
                y_pred = np.argmax(z)
                if y_pred == y:
                    correct += 1
            acc = correct / len(dataset)
            return acc

        # accuracy = calc_accuracy()
        # print("Accuracy:", accuracy)

        a = []
        alcohol = int(request.form['alcohol'])
        malic_acid = int(request.form['malic_acid'])
        ash = int(request.form['ash'])
        alcalinity_of_ash = int(request.form['alcalinity_of_ash'])
        magnesium = int(request.form['magnesium'])
        total_phenols = int(request.form['total_phenols'])
        flavanoids = int(request.form['flavanoids'])
        nonflavanoid_phenols = int(request.form['nonflavanoid_phenols'])
        proanthocyanins = int(request.form['proanthocyanins'])
        colour_intensity = int(request.form['colour_intensity'])
        hue = int(request.form['hue'])
        OD280_OD315_of_diluted_wines = int(request.form['OD280_OD315_of_diluted_wines'])
        proline = int(request.form['proline'])

        a.append(alcohol)
        a.append(malic_acid)
        a.append(ash)
        a.append(alcalinity_of_ash)
        a.append(magnesium)
        a.append(total_phenols)
        a.append(flavanoids)
        a.append(nonflavanoid_phenols)
        a.append(proanthocyanins)
        a.append(colour_intensity)
        a.append(hue)
        a.append(OD280_OD315_of_diluted_wines)
        a.append(proline)

        x = np.array(a)

        def softmax(t):
            out = np.exp(t)
            return out / np.sum(out)

        def predict_otvet(x):
            t1 = x @ W1 + b1
            h1 = relu(t1)
            t2 = h1 @ W2 + b2
            z = softmax(t2)
            return z

        probs = predict_otvet(x)
        pred_class = np.argmax(probs)
        class_names = ['class_0', 'class_1', 'class_2']
        answer = class_names[pred_class]
        return render_template('wine.html', answer=answer)

    return render_template('wine.html')


if __name__ == '__main__':
    app.run()
