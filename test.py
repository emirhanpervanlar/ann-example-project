from tkinter import *
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
import math
import random

def act_sig(x):
    return (1 / (1 + np.exp(-1 * x)))


def er_rmse(target, output):
    error = 0
    count = len(target)
    for i in range(count):
        error = error + (float(target[i]) - output[i]) ** 2
    return np.sqrt((1 / count) * error)


def feedforward(input, weight, bias):
    output = list()
    w_array = np.array(weight)
    for i in range(len(bias)):
        out = 0
        for y in range(len(input)):
            out = out + (float(input[y]) * w_array[y, i])
        out = act_sig(out + bias[i])
        output.append(out)
    return output

def nn_pred():
    epoch = e_count.get()
    momentum = float(m_m.get())
    l_rate = float(l_r.get())
    e_tolerance = float(e_t.get())
    if epoch <= 0:
        messagebox.showinfo("Hata", "Epoch değeri hatalı girildi")
    elif momentum < 0:
        messagebox.showinfo("Hata", "Momentum değeri hatalı girildi.0-1 arası değer giriniz.")
    elif l_rate < 0:
        messagebox.showinfo("Hata", "Öğrenme Katsayısı hatalı girildi.0-1 arası değer giriniz.")
    elif e_tolerance < 0:
        messagebox.showinfo("Hata", "Hata toleransı değeri hatalı girildi.0-1 arası değer giriniz.")
    else:
        if h_u_data.get() == 1:
            input_data = np.load("data/nn_m_input.npy")
            class_data = np.load("data/nn_m_class.npy")
        else:
            input_data = np.load("data/nn_n_input.npy")
            class_data = np.load("data/nn_n_class.npy")
        input_count = len(input_data[0])
        hidden_count = 2
        out_count = 1
        input_data = input_data[0:6]
        target_data = class_data[0:6]
        input_weight_delta = np.zeros((input_count, hidden_count))
        h1_weight_delta = np.zeros((hidden_count, hidden_count))

        if rand_weight.get() == 0:
            input_weight = [[-2.11,0.69],[1.83,1.12],[1.49,1.97]]
            h1_weight = [[-2.89],[1.36]]
            h1_bias = [[0.24,-2.4]]
            output_bias = [[-2.12]]
        else:
            input_weight = np.load("nn_weight/input_weight.npy")
            h1_weight = np.load("nn_weight/h1_weight.npy")
            h1_bias = np.load("nn_weight/h1_bias.npy")
            output_bias = np.load("nn_weight/output_bias.npy")

        h1_bias_delta = np.zeros(hidden_count)
        output_bias_delta = np.zeros(out_count)
        output_error = list()
        acc_array = list()
        true_count = 0
        pre_count = 0
        for j in range(epoch):
            e_true_count = 0
            e_pre_count = 0
            epoch_error = list()
            print("Epoch" + str(j+1))
            for i in range(len(input_data)):
                h1_out = feedforward(input_data[i], input_weight, h1_bias)
                n_out = feedforward(h1_out[0], h1_weight, output_bias)
                n_err = er_rmse(target_data[i], n_out)
                epoch_error.append(n_err)
                # GERİ BESLEME
                if (n_err < e_tolerance):
                    e_true_count += 1
                e_pre_count += 1
                if n_err > e_tolerance:
                    out_err_unit = list()
                    for a in range(len(n_out)):
                         e_o_unit = float(n_out[a]) * (1 - n_out[a]) * (float(target_data[i][a]) - n_out[a])
                         out_err_unit.append(e_o_unit)
                         output_bias_delta[a] = (momentum * e_o_unit) + (l_rate * output_bias_delta[a])
                         output_bias[0][a] = output_bias[0][a] + output_bias_delta[a]
                    h1_err_unit = list()
                    for c in range(len(h1_out[0])):
                        h1_e_unit = 0
                        for xc in range(len(n_out)):
                            h1_e_unit = h1_e_unit + (out_err_unit[xc] * h1_weight[c][xc])
                            h1_weight_delta[c][xc] = (momentum * out_err_unit[xc] * h1_out[0][c]) + (l_rate * h1_weight_delta[c][xc])
                        h1_err_unit.append(h1_out[0][c] * (1 - h1_out[0][c]) * h1_e_unit)
                        h1_bias_delta[c] = (momentum * h1_e_unit) + (l_rate * h1_bias_delta[c])
                        h1_bias[0][c] = h1_bias[0][c] + h1_bias_delta[c]

                    for d in range(len(input_data[i])):
                        for xd in range(len(h1_out[0])):
                            input_weight_delta[d][xd] = (momentum * h1_err_unit[xd] * float(input_data[i][d])) + (l_rate * input_weight_delta[d][xd])

                    # Ağırlık güncelleme
                    for a in range(len(h1_out[0])):
                        for b in range(len(n_out)):
                            h1_weight[a][b] = h1_weight[a][b] + h1_weight_delta[a][b]
                    for a in range(len(input_data[i])):
                        for b in range(len(h1_out[0])):
                            input_weight[a][b] = input_weight[a][b] + input_weight_delta[a][b]

            # Accuracy Hesaplama
            true_count += e_true_count
            pre_count += e_pre_count
            acc_rate = int((true_count / pre_count) * 100)
            ep_rate = int((e_true_count / e_pre_count) * 100)
            print("Epoch True : " + str(e_true_count) + "   Epoch Rate : " + str(ep_rate) + "%  Accuracy :  " + str(acc_rate) + "%")
            output_error.append(epoch_error)
            acc_array.append(acc_rate)

            # Ağırlıkların kaydedilmesi
            np.save("nn_weight/h1_weight", h1_weight)
            np.save("nn_weight/input_weight", input_weight)
            np.save("nn_weight/h1_bias", h1_bias)
            np.save("nn_weight/output_bias", output_bias)

            # Doğruluk oranı kaydı ve grafik olarak gösterme

        plt.plot(acc_array)
        plt.title("Eğitim Sonuçları")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.show()

def nn_test():
    epoch = e_count.get()
    e_tolerance = float(e_t.get())
    if epoch <= 0:
        messagebox.showinfo("Hata", "Epoch değeri hatalı girildi")
    elif e_tolerance < 0:
        messagebox.showinfo("Hata", "Hata toleransı değeri hatalı girildi.0-1 arası değer giriniz.")
    else:
        if h_u_data.get() == 1:
            input_data = np.load("data/nn_m_input.npy")
            class_data = np.load("data/nn_m_class.npy")
        else:
            input_data = np.load("data/nn_n_input.npy")
            class_data = np.load("data/nn_n_class.npy")
        input_data = input_data[4:8]
        target_data = class_data[4:8]
        input_weight = np.load("nn_weight/input_weight.npy")
        h1_weight = np.load("nn_weight/h1_weight.npy")
        h1_bias = np.load("nn_weight/h1_bias.npy")
        output_bias = np.load("nn_weight/output_bias.npy")
        output_error = list()
        acc_array = list()
        true_count = 0
        pre_count = 0
        for j in range(epoch):
            e_true_count = 0
            e_pre_count = 0
            epoch_error = list()
            print("Epoch" + str(j+1))
            for i in range(len(input_data)):
                h1_out = feedforward(input_data[i], input_weight, h1_bias)
                n_out = feedforward(h1_out[0], h1_weight, output_bias)
                n_err = er_rmse(target_data[i], n_out)
                epoch_error.append(n_err)
                # GERİ BESLEME
                if (n_err < e_tolerance):
                    e_true_count += 1
                e_pre_count += 1
            true_count += e_true_count
            pre_count += e_pre_count
            acc_rate = int((true_count / pre_count) * 100)
            ep_rate = int((e_true_count / e_pre_count) * 100)
            print("Epoch True : " + str(e_true_count) + "   Epoch Rate : " + str(ep_rate) + "%  Accuracy :  " + str(acc_rate) + "%")
            output_error.append(epoch_error)
            acc_array.append(acc_rate)
        plt.plot(acc_array)
        plt.title("Test Sonuçları")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.show()

def draw_neural_net():
    fig = plt.figure(figsize=(7, 5))
    ax = fig.gca()
    ax.axis('off')
    left = .1
    right = .8
    bottom = .3
    top = .7
    layer_sizes = [3,2,1]
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom) / float(max(layer_sizes))
    h_spacing = (right - left) / float(len(layer_sizes) - 1)
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2. + (top + bottom) / 2.
        for m in range(layer_size):
            circle = plt.Circle((n * h_spacing + left, layer_top - m * v_spacing), v_spacing / 4.,
                                color='w', ec='k', zorder=4)
            ax.add_artist(circle)
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing * (layer_size_a - 1) / 2. + (top + bottom) / 2.
        layer_top_b = v_spacing * (layer_size_b - 1) / 2. + (top + bottom) / 2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n * h_spacing + left, (n + 1) * h_spacing + left],
                                  [layer_top_a - m * v_spacing, layer_top_b - o * v_spacing], c='k')
                ax.add_artist(line)
    plt.show()


if __name__ == "__main__":
    master = Tk()
    master.title("Makina Öğrenmesi Projesi")
    master.geometry("500x400")
    h_u_data = IntVar()
    u_data = Checkbutton(master, text="Verileri Karışık Kullan",variable=h_u_data)
    u_data.pack()

    rand_weight = IntVar()
    r_w = Checkbutton(master, text="Eğitimde Kayıtlı Ağırlıkları Kullan", variable=rand_weight)
    r_w.pack()

    lb1 = Label(master,text="Epoch Sayısı")
    lb1.pack()
    e_count = IntVar(value=10)
    e_c = Entry(master,textvariable=e_count)
    e_c.pack()

    lb2 = Label(master, text="Öğrenme Katsayısı")
    lb2.pack()
    l_rate = IntVar(value=0.3)
    l_r = Entry(master, textvariable=l_rate)
    l_r.pack()

    lb3 = Label(master, text="Momentum Katsayısı")
    lb3.pack()
    mome = IntVar(value=0.3)
    m_m = Entry(master, textvariable=mome)
    m_m.pack()

    lb4 = Label(master, text="Hata Toleransı")
    lb4.pack()
    error_t = IntVar(value=0.1)
    e_t = Entry(master, textvariable=error_t)
    e_t.pack()

    education_btn = Button(text="Ağı Eğit", command=nn_pred)
    education_btn.pack(side="top")

    test_btn = Button(text="Ağı Test Et", command=nn_test)
    test_btn.pack(side="top")

    draw_btn = Button(text="Ağı Çiz", command=draw_neural_net)
    draw_btn.pack(side="top")
    mainloop()