import numpy as np
import csv, os

test_path = os.path.join("MNIST_CSV", "mnist_test.csv")
train_path = os.path.join("MNIST_CSV", "mnist_train.csv")

train_data = []

with open(train_path, "r") as file:
    csvreader = csv.reader(file)
    for idx, row in enumerate(csvreader):
        train_data.append([int(el) for el in row])

train_data = train_data

test_data = []

with open(test_path, "r") as file:
    csvreader = csv.reader(file)
    for idx, row in enumerate(csvreader):
        test_data.append([int(el) for el in row])

# Функция активации ReLU
def relu(x):
    return np.maximum(x, 0)

# Функция softmax
def softmax(x):
    exp_values = np.exp(x - np.max(x))
    return exp_values / np.sum(exp_values)

def conv_forward(X, W):
    """
    Выполняет свертку изображения с заданными фильтрами.

    Параметры:
    X (numpy array): Входное изображение.
    W (numpy array): Фильтры свертки.
    b (numpy array): Смещения.

    Возвращает:
    numpy array: Результат свертки.
    """
    input_height, input_width = X.shape
    num_filters, filter_height, filter_width = W.shape

    # Размеры результата свертки
    output_height = input_height - filter_height + 1
    output_width = input_width - filter_width + 1

    # Результат свертки
    Z = np.zeros((output_height, output_width, num_filters))

    # Применение свертки
    for k in range(num_filters):
        for i in range(output_height):
            for j in range(output_width):
                Z[i, j, k] = np.sum(X[i:i+filter_height, j:j+filter_width] * W[k])

    return Z

def max_pool_forward(X, pool_size=2, stride=2):
    """
    Выполняет операцию MaxPooling на входных данных.

    Параметры:
    X (numpy array): Входные данные.
    pool_size (int): Размер окна операции MaxPooling (по умолчанию 2).
    stride (int): Шаг операции MaxPooling (по умолчанию 2).

    Возвращает:
    numpy array: Результат операции MaxPooling.
    """
    input_height, input_width, num_filters = X.shape

    # Размеры результата MaxPooling
    output_height = (input_height - pool_size) // stride + 1
    output_width = (input_width - pool_size) // stride + 1

    # Результат MaxPooling
    Z = np.zeros((output_height, output_width, num_filters))

    # Применение MaxPooling
    for i in range(output_height):
        for j in range(output_width):
            X_slice = X[i * stride:i * stride + pool_size, j * stride:j * stride + pool_size, :]
            Z[i, j, :] = np.max(X_slice, axis=(0, 1))

    return Z

def init_weights(num_classes):
    kernels = np.random.randn(8, 3, 3)
    W_dense = np.random.randn(13*13*8, num_classes)
    return kernels, W_dense

def normalize(x):
    x = np.array(x)
    return x/255

def max_pool_backward(grad_A, A, pool_size=2, stride=2):
    input_height, input_width, num_filters = A.shape
    
    # Размеры результата MaxPooling
    output_height = (input_height - pool_size) // stride + 1
    output_width = (input_width - pool_size) // stride + 1
    
    # Инициализация градиента по входу
    grad_A_prev = np.zeros_like(A)
    
    for i in range(output_height):
        for j in range(output_width):
            for k in range(num_filters):
                # Находим максимальное значение и его индекс в окне
                x_start, x_end = i * stride, i * stride + pool_size
                y_start, y_end = j * stride, j * stride + pool_size
                window = A[x_start:x_end, y_start:y_end, k]
                max_val = np.max(window)
                
                # Находим индекс максимального значения
                max_index = np.argmax(window)
                
                # Находим координаты максимального значения в окне
                max_i, max_j = np.unravel_index(max_index, window.shape)
                
                # Распространяем градиент только через ячейку, соответствующую максимальному значению
                grad_A_prev[x_start + max_i, y_start + max_j, k] = grad_A[i, j, k]
    
    return grad_A_prev


def make_windows(image, size):
    """
    Makes slices from image using NxN windows
    """
    rows, cols = image.shape
    result = [] 
    for i in range(rows-2):
        for j in range(cols-2):
            # Extracting a NxN patch from the image
            patch = image[i:i+size, j:j+size]
            result.append(patch)  
    return np.array(result)


num_classes = 10
kernels, W1 = init_weights(num_classes)

def train(data, num_epochs, learning_rate, kernels, W1):
    for epoch in range(num_epochs):
        total_accuracy = 0
        for i,el in enumerate(data):
            Y = np.zeros(10)
            Y[el[0]] = 1
            x = np.array(el[1:])
            x_prepeared = normalize(x)
            """Forward prop"""
            Z1 = conv_forward(x_prepeared.reshape((28,28)), kernels) #(26, 26, 8)
            A1 = relu(Z1) #(26, 26, 8)
            Z2 = max_pool_forward(A1) #(13, 13)
            Z2_flat = Z2.flatten()
            Z3 = np.dot(Z2_flat,W1)
            predicted_probs = softmax(Z3)
            """losses"""
            loss = -np.sum(Y * np.log(predicted_probs))
            accuracy = 1 if np.argmax(predicted_probs) == np.argmax(Y) else 0
            total_accuracy+=accuracy
            """Backward prop"""
            dZ3 = predicted_probs - Y #10x1 --- Oшибка на выходном слое
            dW1 = np.outer(Z2_flat, dZ3) #1352x10
            dZ2_flat = np.dot(dZ3, W1.T) #1352x1 Ошибка на входе в ПС (МаксПулл)
            dZ2 = dZ2_flat.reshape(Z2.shape) #13x13x8 - решейп
            dA1 = max_pool_backward(dZ2, A1) #26x26x8 - Градиенты до МаксПулл
            dZ1 = dA1 * (A1 > 0) #deriv of ReLU 26x26x8 - Градиенты до Активации
            dZ1 = dZ1.reshape((676,8))
            # padded_dZ1 = np.pad(dZ1, pad_width=((1, 1), (1, 1), (0, 0)), mode='constant') #28x28x8
            # padded_dZ1 = padded_dZ1.reshape((28*28,8)) #784x8 - Градиенты для Ядер
            X_sliced = make_windows(x_prepeared.reshape((28,28)),3)
            X_sliced = X_sliced.reshape((676,9)) #676x9
            k_update = np.dot(X_sliced.T, dZ1) #9x8
            k_update = k_update.reshape((3,3,8)).T
            """Weights update (SDG)"""
            W1 -= learning_rate * dW1
            kernels -= learning_rate * k_update

            if i % 1 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Sample {i}/{len(data)}, Loss: {loss}, Accuracy: {total_accuracy/((i)+1)}")

def max_index(x):
    max_x = max(x[0])
    for i, el in enumerate(x[0]):
        if el == max_x:
            return i
        
result = {"Corr":0, "Err":0}
        
def test(test_data, kernels, W1):
    total_accuracy = 0
    for i,el in enumerate(test_data):
        Y = np.zeros(10)
        Y[el[0]] = 1
        x = np.array(el[1:])
        x_prepeared = normalize(x)
        """Forward prop"""
        Z1 = conv_forward(x_prepeared.reshape((28,28)), kernels) #(26, 26, 8)
        A1 = relu(Z1) #(26, 26, 8)
        Z2 = max_pool_forward(A1) #(13, 13)
        Z2_flat = Z2.flatten()
        Z3 = np.dot(Z2_flat,W1)
        predicted_probs = softmax(Z3)
        num = max_index(predicted_probs)
        if num == el[0]:
            result["Corr"] +=1
        else:
            result["Err"] += 1

    print(result)



train(train_data, num_epochs=5, learning_rate=0.1, kernels=kernels, W1=W1)
test(train_data, kernels,W1)
