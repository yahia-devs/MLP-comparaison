# MLP from scratch
# Yahia Chemlali
#
# Le but c'est de coder un perceptron multi couches sans pytorch/tensorflow,
# juste numpy. On le teste sur Fashion-MNIST : 70 000 images 28x28 de
# vetements (t-shirts, chaussures, sacs...), 10 classes.
# L'important c'est de comprendre les maths derrière : fonction d'activation,
# backpropagation, etc.
#
# ref : https://web.stanford.edu/~jurafsky/slp3/

#      : https://rtavenar.github.io/deep_book/book_fr.pdf

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


CLASS_NAMES = [
    'T-shirt', 'Pantalon', 'Pull', 'Robe', 'Manteau',
    'Sandale', 'Chemise', 'Basket', 'Sac', 'Bottine'
]


def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    # return z > 0
    return (z > 0).astype(float)

def softmax(z):
    # e = np.exp(z)
    e = np.exp(z - np.max(z, axis=0, keepdims=True))
    return e / np.sum(e, axis=0, keepdims=True)



def cross_entropy(pred, vrai):
    m = vrai.shape[1]
    return -np.sum(vrai * np.log(pred + 1e-12)) / m




class MLP:
    def __init__(self, couches, acts, lr=0.01):
        self.acts = acts
        self.lr = lr
        self.L = len(couches) - 1
        rng = np.random.default_rng(42)
        self.W, self.b = {}, {}
        for i in range(1, self.L + 1):
            n_in, n_out = couches[i-1], couches[i]
            scale = np.sqrt(2.0/n_in) if acts[i-1] == 'relu' else np.sqrt(1.0/n_in)
            self.W[i] = rng.normal(0, scale, (n_out, n_in))
            self.b[i] = np.zeros((n_out, 1))

    def forward(self, X):
        mem = {'A0': X}
        A = X
        for i in range(1, self.L + 1):
            Z = self.W[i] @ A + self.b[i]
            mem['Z' + str(i)] = Z
            A = softmax(Z) if self.acts[i-1] == 'softmax' else relu(Z)
            mem['A' + str(i)] = A
        return A, mem

    def backward(self, y, mem):
        m = y.shape[1]
        grads = {}

        ## dZ = mem['A' + str(self.L)] - y
        ## grads['dW' + str(self.L)] = dZ @ mem['A' + str(self.L-1)].T
        ## grads['db' + str(self.L)] = np.sum(dZ, axis=1, keepdims=True)

        dZ = mem['A' + str(self.L)] - y
        grads['dW' + str(self.L)] = (1/m) * dZ @ mem['A' + str(self.L-1)].T
        grads['db' + str(self.L)] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        for i in range(self.L-1, 0, -1):
            dA = self.W[i+1].T @ dZ
            dZ = dA * relu_deriv(mem['Z' + str(i)])
            grads['dW' + str(i)] = (1/m) * dZ @ mem['A' + str(i-1)].T
            grads['db' + str(i)] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        return grads

    def update(self, grads):
        for i in range(1, self.L + 1):
            self.W[i] -= self.lr * grads['dW' + str(i)]
            self.b[i] -= self.lr * grads['db' + str(i)]

    def train(self, X, y, iterations=30, batch_size=64, log=5):
        m = X.shape[1]
        losses = []
        accs = []
        for it in range(1, iterations+1):
            idx = np.random.permutation(m)
            batch_loss = 0
            nb = 0
            for start in range(0, m, batch_size):
                batch = idx[start:start+batch_size]
                Xb, yb = X[:, batch], y[:, batch]
                out, mem = self.forward(Xb)
                batch_loss += cross_entropy(out, yb)
                nb += 1
                grads = self.backward(yb, mem)
                self.update(grads)
            avg_loss = batch_loss / nb
            losses.append(avg_loss)
            out_full, _ = self.forward(X)
            acc = np.mean(np.argmax(out_full, 0) == np.argmax(y, 0))
            accs.append(acc)
            if it % log == 0 or it == 1:
                print("iteration " + str(it) + "  loss=" + str(round(avg_loss, 4)) + "  acc=" + str(round(acc * 100, 2)) + "%")
        return losses, accs

    def predict(self, X):
        out, _ = self.forward(X)
        return np.argmax(out, axis=0)


def one_hot(y, k):
    # oh = np.eye(k)[y]
    oh = np.zeros((k, len(y)))
    oh[y, np.arange(len(y))] = 1
    return oh




print("chargement de Fashion-MNIST...")
mnist = fetch_openml('Fashion-MNIST', version=1, as_frame=False)
X = mnist.data / 255.0
y = mnist.target.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test = X_train.T, X_test.T

y_train_oh = one_hot(y_train, 10)
y_test_oh  = one_hot(y_test, 10)

print("train: " + str(X_train.shape[1]) + " samples")
print("test:  " + str(X_test.shape[1]) + " samples")
print("features: " + str(X_train.shape[0]))

fig, axes = plt.subplots(2, 8, figsize=(12, 3))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[:, i].reshape(28, 28), cmap='gray')
    ax.set_title(CLASS_NAMES[y_train[i]], fontsize=7)
    ax.axis('off')
plt.suptitle('exemples du dataset')
plt.tight_layout()
plt.show()


net = MLP(
    couches=[784, 256, 128, 10],
    acts=['relu', 'relu', 'softmax'],
    lr=0.1
)

hist, accs = net.train(X_train, y_train_oh, iterations=30, batch_size=64, log=5)

pred_test = net.predict(X_test)
acc_test = np.mean(pred_test == y_test)
print("accuracy sur le test set: " + str(round(acc_test * 100, 2)) + "%")

fig, ax1 = plt.subplots(figsize=(9, 4))

ax1.plot(hist, color='steelblue', label='loss')
ax1.set_xlabel('iteration')
ax1.set_ylabel('loss', color='steelblue')
ax1.tick_params(axis='y', labelcolor='steelblue')

ax2 = ax1.twinx()
ax2.plot(accs, color='coral', label='accuracy')
ax2.set_ylabel('accuracy', color='coral')
ax2.tick_params(axis='y', labelcolor='coral')

fig.legend(loc='center right', bbox_to_anchor=(0.88, 0.5))
plt.title('loss et accuracy par iteration')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

cm = confusion_matrix(y_test, pred_test)
fig, ax = plt.subplots(figsize=(10, 8))
ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=ax, cmap='Blues')
plt.title('matrice de confusion')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(2, 8, figsize=(12, 3))
for i, ax in enumerate(axes.flat):
    img = X_test[:, i].reshape(28, 28)
    ax.imshow(img, cmap='gray')
    p = pred_test[i]
    v = y_test[i]
    color = 'green' if p == v else 'red'
    ax.set_title(CLASS_NAMES[p], color=color, fontsize=7)
    ax.axis('off')
plt.suptitle('vert=bon, rouge=erreur')
plt.tight_layout()
plt.show()

