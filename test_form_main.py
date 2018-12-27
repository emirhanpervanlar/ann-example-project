import tkinter as tk
import numpy as np
import math
import random
from tkinter import messagebox,IntVar
import matplotlib.pyplot as plt

class Application(tk.Frame):
    input_data = np.load("data/nn_m_input.npy")
    class_data = np.load("data/nn_m_class.npy")
    input_count = len(input_data[0])
    hidden_count = 2
    out_count = 1
    momentum = 0.3
    l_rate = 0.3
    epoch = 10
    e_tolerance = 0.1
    def __init__(self, master=None):
        super().__init__(master)
        self.pack()
        self.create_widgets()



    def create_widgets(self):

        self.education_btn = tk.Button(self, text="Eğitim", command=self.nn_pred)
        self.education_btn.pack(side="top")

        self.test_btn = tk.Button(self, text="Test", command=self.nn_test)
        self.test_btn.pack(side="top")

        v = IntVar()
        self.edu_check = tk.Checkbutton(self, text="male", variable=v)
        self.edu_check.pack(side="top")
        self.edu_check.var = v


        self.quit = tk.Button(self, text="QUIT", fg="red",command=root.destroy)
        self.quit.pack(side="bottom")


    def act_sig(self,x):
        return (1 / (1 + np.exp(-1 * x)))

    def er_rmse(self,target, output):
        error = 0
        count = len(target)
        for i in range(count):
            error = error + (float(target[i]) - output[i]) ** 2
        return np.sqrt((1 / count) * error)

    def feedforward(self,input, weight, bias):
        output = list()
        w_array = np.array(weight)
        for i in range(len(bias)):
            out = 0
            for y in range(len(input)):
                out = out + (float(input[y]) * w_array[y, i])
            out = self.act_sig(out + bias[i])
            output.append(out)
        return output

    def nn_pred(self):
        input_data = self.input_data[0:6]
        target_data = self.class_data[0:6]
        input_weight_delta = np.zeros((self.input_count, self.hidden_count))
        h1_weight_delta = np.zeros((self.hidden_count, self.hidden_count))
        input_weight = [[-2.11,0.69],[1.83,1.12],[1.49,1.97]]
        h1_weight = [[-2.89],[1.36]]
        h1_bias = [[0.24,-2.4]]
        output_bias = [[-2.12]]
        h1_bias_delta = np.zeros(self.hidden_count)
        output_bias_delta = np.zeros(self.out_count)
        output_error = list()
        acc_array = list()
        true_count = 0
        pre_count = 0

        for j in range(self.epoch):
            e_true_count = 0
            e_pre_count = 0
            epoch_error = list()
            print("Epoch" + str(j))
            for i in range(len(input_data)):
                h1_out = self.feedforward(input_data[i], input_weight, h1_bias)
                n_out = self.feedforward(h1_out[0], h1_weight, output_bias)
                n_err = self.er_rmse(target_data[i], n_out)
                epoch_error.append(n_err)

                # GERİ BESLEME
                if (n_err < self.e_tolerance):
                    e_true_count += 1
                e_pre_count += 1
                # print("İterasyon_ "+str(i)+"  Hata : "+str(n_err))
                if n_err > self.e_tolerance:
                    out_err_unit = list()
                    for a in range(len(n_out)):
                        e_o_unit = float(n_out[a]) * (1 - n_out[a]) * (float(target_data[i][a]) - n_out[a])
                        out_err_unit.append(e_o_unit)
                        output_bias_delta[a] = (self.momentum * e_o_unit) + (self.l_rate * output_bias_delta[a])
                        output_bias[0][a] = output_bias[0][a] + output_bias_delta[a]

                    h1_err_unit = list()
                    for c in range(len(h1_out[0])):
                        h1_e_unit = 0
                        for xc in range(len(n_out)):
                            h1_e_unit = h1_e_unit + (out_err_unit[xc] * h1_weight[c][xc])
                            h1_weight_delta[c][xc] = (self.momentum * out_err_unit[xc] * h1_out[0][c]) + (self.l_rate * h1_weight_delta[c][xc])
                        h1_err_unit.append(h1_out[0][c] * (1 - h1_out[0][c]) * h1_e_unit)
                        h1_bias_delta[c] = (self.momentum * h1_e_unit) + (self.l_rate * h1_bias_delta[c])
                        h1_bias[0][c] = h1_bias[0][c] + h1_bias_delta[c]

                    for d in range(len(input_data[i])):
                        for xd in range(len(h1_out[0])):
                            input_weight_delta[d][xd] = (self.momentum * h1_err_unit[xd] * float(input_data[i][d])) + (self.l_rate * input_weight_delta[d][xd])

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



    def nn_test(self):
        input_data = self.input_data[4:8]
        target_data = self.class_data[4:8]
        input_weight = np.load("nn_weight/input_weight.npy")
        h1_weight = np.load("nn_weight/h1_weight.npy")
        h1_bias = np.load("nn_weight/h1_bias.npy")
        output_bias = np.load("nn_weight/output_bias.npy")
        output_error = list()
        acc_array = list()
        true_count = 0
        pre_count = 0
        for j in range(self.epoch):
            e_true_count = 0
            e_pre_count = 0
            epoch_error = list()
            for i in range(len(input_data)):
                h1_out = self.feedforward(input_data[i], input_weight, h1_bias)
                n_out = self.feedforward(h1_out[0], h1_weight, output_bias)
                n_err = self.er_rmse(target_data[i], n_out)
                epoch_error.append(n_err)
                # GERİ BESLEME
                if (n_err < self.e_tolerance):
                    e_true_count += 1
                e_pre_count += 1
            true_count += e_true_count
            pre_count += e_pre_count
            acc_rate = int((true_count / pre_count) * 100)
            ep_rate = int((e_true_count / e_pre_count) * 100)
            print("Epoch True : " + str(e_true_count) + "   Epoch Rate : " + str(ep_rate) + "%  Accuracy :  " + str(
                acc_rate) + "%")
            output_error.append(epoch_error)
            acc_array.append(acc_rate)
        plt.plot(acc_array)
        plt.title("Eğitim Sonuçları")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.show()


root = tk.Tk()
root.title("Makine Öğrenmesi Projesi")
root.geometry("400x200")
app = Application(master=root)
app.mainloop()