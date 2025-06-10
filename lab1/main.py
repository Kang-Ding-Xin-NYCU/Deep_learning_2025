import numpy as np
import generate_data as gd

def sig(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def cross_entropy(label, output):
    output = np.clip(output, 1e-5, 1 - 1e-5)
    return -(label * np.log(output) + (1 - label) * np.log(1 - output))

def sig_derivative(s):
    x = sig(s)
    return x * (1 - x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def cross_entropy_derivative(label, output):
    output = np.clip(output, 1e-5, 1 - 1e-5)
    return -(label/output)+((1 - label)/(1 - output))

def conv1d(x, W, B):
    num_filters, kernel_size = W.shape
    output_length = len(x) - kernel_size + 1
    output = np.zeros((num_filters, output_length))

    for f in range(num_filters):
        for i in range(output_length):
            output[f, i] = np.sum(x[i:i+kernel_size] * W[f]) + B[f]

    return output.T

def forward(x, activation, conv, W1, W2, W3, B1, B2, B3):
    if activation == "sigmoid":
        if conv == True:
            mat1 = conv1d(x, W1, B1)
        else:
            mat1 = np.matmul(x, W1) + B1
        act1 = sig(mat1)
        mat2 = np.matmul(act1, W2) + B2
        act2 = sig(mat2)
    elif activation == "tanh":
        if conv == True:
            mat1 = conv1d(x, W1, B1)
        else:
            mat1 = np.matmul(x, W1) + B1
        act1 = tanh(mat1)
        mat2 = np.matmul(act1, W2) + B2
        act2 = tanh(mat2)

    output = np.matmul(act2, W3) + B3
    output_act = sig(output)

    return mat1, act1, mat2,  act2, output, output_act

def train(datas, labels, hidden_layer, learning_rate, activation, optimizer, epoch, conv, filters, kernels):

    if conv == True:
        W1 = np.random.randn(filters, kernels) 
        B1 = np.random.randn(filters)
        W2 = np.random.randn(filters, hidden_layer)
        vW1 = np.zeros((filters, kernels))
        vB1 = np.zeros((filters))
        vW2 = np.zeros((filters, hidden_layer)) 
    else:
        W1 = np.random.randn(2, hidden_layer) 
        B1 = np.random.randn(1, hidden_layer)
        W2 = np.random.randn(hidden_layer, hidden_layer)
        vW1 = np.zeros((2, hidden_layer))
        vB1 = np.zeros((1, hidden_layer))
        vW2 = np.zeros((hidden_layer, hidden_layer)) 
    
    W3 = np.random.randn(hidden_layer, 1) 
    B2 = np.random.randn(1, hidden_layer)
    B3 = np.random.randn(1, 1)
    vW3 = np.zeros((hidden_layer, 1))
    vB2 = np.zeros((1, hidden_layer))
    vB3 = np.zeros((1, 1))
    beta = 0.75


    epoch_losses = []
    for i in range(1, epoch + 1):
        batch_loss = []
        for j in range(datas.shape[0]):
            x = datas[j]
            y = labels[j]
            mat1, act1, mat2, act2, output, output_act = forward(x, activation, conv, W1, W2, W3, B1, B2, B3)
            loss = cross_entropy(y, output_act)
            batch_loss.append(np.mean(loss))

            if activation == "sigmoid":
                dL3 = sig_derivative(output) * cross_entropy_derivative(y, output_act)
                dW3 = np.matmul(act2.T, dL3)
                dB3 = np.sum(dL3, axis=0, keepdims=True)
                dL2 = np.matmul(dL3, W3.T) * sig_derivative(mat2)
                dW2 = np.matmul(act1.T, dL2)
                dB2 = np.sum(dL2, axis=0, keepdims=True)
                dL1 = np.matmul(dL2, W2.T) * sig_derivative(mat1)
                if conv == True:
                    dW1 = np.zeros_like(W1)
                    dB1 = np.zeros_like(B1)
                    for k in range(filters):
                        for l in range(kernels):
                            dW1[k, l] = np.sum(dL1.T[k] * x[l:l + len(dL1.T[k])])
                        dB1[k] = np.sum(dL1.T[k])
                else:
                    dW1 = np.matmul(x.reshape(1, -1).T, dL1)
                    dB1 = np.sum(dL1, axis=0, keepdims=True)
            elif activation == "tanh":
                dL3 = sig_derivative(output) * cross_entropy_derivative(y, output_act)
                dW3 = np.matmul(act2.T, dL3)
                dB3 = np.sum(dL3, axis=0, keepdims=True)
                dL2 = np.matmul(dL3, W3.T) * tanh_derivative(mat2)
                dW2 = np.matmul(act1.T, dL2)
                dB2 = np.sum(dL2, axis=0, keepdims=True)
                dL1 = np.matmul(dL2, W2.T) * tanh_derivative(mat1)
                if conv == True:
                    dW1 = np.zeros_like(W1)
                    dB1 = np.zeros_like(B1)
                    for k in range(filters):
                        for l in range(kernels):
                            dW1[k, l] = np.sum(dL1.T[k] * x[l:l + len(dL1.T[k])])
                        dB1[k] = np.sum(dL1.T[k])
                else:
                    dW1 = np.matmul(x.reshape(1, -1).T, dL1)
                    dB1 = np.sum(dL1, axis=0, keepdims=True)

            if optimizer == "SGD":
                W3 -= learning_rate * dW3
                W2 -= learning_rate * dW2
                W1 -= learning_rate * dW1
                B3 -= learning_rate * dB3
                B2 -= learning_rate * dB2
                B1 -= learning_rate * dB1
            elif optimizer == "Momentum":
                vW3 = beta * vW3 - learning_rate * dW3
                vW2 = beta * vW2 - learning_rate * dW2
                vW1 = beta * vW1 - learning_rate * dW1
                vB3 = beta * vB3 - learning_rate * dB3
                vB2 = beta * vB2 - learning_rate * dB2
                vB1 = beta * vB1 - learning_rate * dB1
                W3 += vW3
                W2 += vW2
                W1 += vW1
                B3 += vB3
                B2 += vB2
                B1 += vB1
        
        epoch_losses.append(np.mean(batch_loss))

        if i % 100 == 0:
            print(f"{i} epoch loss : {np.mean(epoch_losses):.9f}")

    return W1, W2, W3, B1, B2, B3, epoch_losses

def test(datas, labels, activation, conv, W1, W2, W3, B1, B2, B3):  
    count = 0
    outputs = np.zeros((datas.shape[0], 1))
    for x , y in zip(datas, labels):
        _, _, _, _, _, output_act = forward(x, activation, conv, W1, W2, W3, B1, B2, B3)
        outputs[count, 0] = np.round(output_act.item())
        count += 1
        print(f"Iter{count} |  Grount truth {labels[count - 1][0]} | prrediction: {output_act[0][0]}")

    acc = np.sum(outputs == labels) / outputs.shape[0]

    print(f"acc : {acc * 100:.3f}%")
    return outputs

def plot(datas, labels, outputs, losses, name):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(range(1, len(losses) + 1), losses, label="train_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("Learning curve with " + name + ".png")

    plt.figure()
    point_gt = np.concatenate((datas, labels), axis=1)
    class0_gt = point_gt[point_gt[:, 2] == 0]
    class1_gt = point_gt[point_gt[:, 2] == 1]

    point_pred = np.concatenate((datas, outputs), axis=1)
    class0_pred = point_pred[point_pred[:, 2] == 0]
    class1_pred = point_pred[point_pred[:, 2] == 1]

    _, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].scatter(class0_gt[:, 0], class0_gt[:, 1], c='blue', label='Class 0')
    axs[0].scatter(class1_gt[:, 0], class1_gt[:, 1], c='red', label='Class 1')
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].set_title("Ground truth")
    axs[0].legend()

    axs[1].scatter(class0_pred[:, 0], class0_pred[:, 1], c='blue', label='Class 0')
    axs[1].scatter(class1_pred[:, 0], class1_pred[:, 1], c='red', label='Class 1')
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("y")
    axs[1].set_title("Predict result")
    axs[1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle(f"Comparison: {name}", fontsize=14, fontweight='bold')
    plt.savefig(f"GT_vs_Pred_{name}.png", bbox_inches="tight")


def main():
    datas1, labels1 = gd.generate_XOR_easy()
    W1, W2, W3, B1, B2, B3, epoch_losses = train(datas1, labels1, hidden_layer = 256, learning_rate = 0.001, activation = "sigmoid", optimizer = "SGD" ,      epoch = 1000, conv = True, filters = 8, kernels = 2)
    outputs = test(datas1, labels1, "sigmoid", True, W1, W2, W3, B1, B2, B3)
    plot(datas1, labels1, outputs, epoch_losses, name = "XOR_sigmoid_SGD")
    datas2, labels2 = gd.generate_XOR_easy()
    W1, W2, W3, B1, B2, B3, epoch_losses = train(datas2, labels2, hidden_layer = 256, learning_rate = 0.001, activation = "sigmoid", optimizer = "Momentum" , epoch = 1000, conv = True, filters = 8, kernels = 2)
    outputs = test(datas2, labels2, "sigmoid", True, W1, W2, W3, B1, B2, B3)
    plot(datas2, labels2, outputs, epoch_losses, name = "XOR_sigmoid_Momentum")
    datas3, labels3 = gd.generate_XOR_easy()
    W1, W2, W3, B1, B2, B3, epoch_losses = train(datas3, labels3, hidden_layer = 256, learning_rate = 0.001, activation = "tanh",    optimizer = "SGD",       epoch = 1000, conv = True, filters = 8, kernels = 2)
    outputs = test(datas3, labels3, "tanh", True, W1, W2, W3, B1, B2, B3)
    plot(datas3, labels3, outputs, epoch_losses, name = "XOR_tanh_SGD")
    datas4, labels4 = gd.generate_XOR_easy()
    W1, W2, W3, B1, B2, B3, epoch_losses = train(datas4, labels4, hidden_layer = 256, learning_rate = 0.001, activation = "tanh",    optimizer = "Momentum",  epoch = 1000, conv = True, filters = 8, kernels = 2)
    outputs = test(datas4, labels4, "tanh", True, W1, W2, W3, B1, B2, B3)
    plot(datas4, labels4, outputs, epoch_losses, name = "XOR_tanh_Momentum")

if __name__ == '__main__':
    main()
