from tkinter import *
import numpy as np
from tkinter import messagebox
import matplotlib.pyplot as plt
import math
import random
import pandas as pd
import os


def printSelection(i):
        if int(vars[i].get()) == 1:
                select_arr.append("%s"%(i))
        else:
            xx = select_arr.index("%s"%(i))
            del(select_arr[xx])

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


def nn_edu():
    if len(select_arr) == 0:
        messagebox.showinfo("Hata", "Seçili örnek bulunamadı")
    elif float(l_r.get()) < 0 or float(l_r.get()) > 1:
        messagebox.showinfo("Hata", "Öğrenme katsayı değeri 0-1 aralığında veya eşit olmalıdır.")
    elif float(m_m.get()) < 0 or float(m_m.get()) > 1:
        messagebox.showinfo("Hata", "Momentum katsayı değeri 0-1 aralığında veya eşit olmalıdır.")
    elif int(e_c.get()) < 0:
        messagebox.showinfo("Hata", "Epoch değeri 0 ve 0'dan küçük olamaz.")
    elif float(e_t.get()) < 0 or float(e_t.get()) > 1:
        messagebox.showinfo("Hata", "Hata katsayı değeri 0-1 aralığında veya eşit olmalıdır.")
    else:
        epoch = int(e_c.get())
        momentum = float(m_m.get())
        l_rate = float(l_r.get())
        e_tolerance = float(e_t.get())
        input_data = list()
        class_data = list()
        for i in range(len(select_arr)):
            input_data.append(all_data[int(select_arr[i])][0:3])
            class_data.append([all_data[int(select_arr[i])][3][0]])
        print(input_data)
        print(class_data)
        input_count = len(input_data[0])
        hidden_count = 2
        out_count = 1
        target_data = class_data
        input_weight_delta = np.zeros((input_count, hidden_count))
        h1_weight_delta = np.zeros((hidden_count, hidden_count))

        if use_weight.get() == 0:
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
        all_epoch_weight = list()
        all_it_weight = list()
        for j in range(epoch):

            e_true_count = 0
            e_pre_count = 0
            epoch_error = list()

            if show_e_weight.get() == 1:
                epoch_weight = list()
                for i in range(len(input_weight)):
                    for j in range(len(input_weight[i])):
                        epoch_weight.append(input_weight[i][j])
                epoch_weight.append(h1_weight[0][0])
                epoch_weight.append(h1_weight[1][0])
                epoch_weight.append(h1_bias[0][0])
                epoch_weight.append(h1_bias[0][1])
                epoch_weight.append(output_bias[0][0])
                all_epoch_weight.append(epoch_weight)


            for i in range(len(input_data)):
                if show_i_weight.get() == 1:
                    it_weight = list()
                    for m in range(len(input_weight)):
                        for n in range(len(input_weight[m])):
                            it_weight.append(input_weight[m][n])
                    it_weight.append(h1_weight[0][0])
                    it_weight.append(h1_weight[1][0])
                    it_weight.append(h1_bias[0][0])
                    it_weight.append(h1_bias[0][1])
                    it_weight.append(output_bias[0][0])
                    all_it_weight.append(it_weight)
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
                            h1_weight_delta[c][xc] = (momentum * out_err_unit[xc] * h1_out[0][c]) + (
                                        l_rate * h1_weight_delta[c][xc])
                        h1_err_unit.append(h1_out[0][c] * (1 - h1_out[0][c]) * h1_e_unit)
                        h1_bias_delta[c] = (momentum * h1_e_unit) + (l_rate * h1_bias_delta[c])
                        h1_bias[0][c] = h1_bias[0][c] + h1_bias_delta[c]

                    for d in range(len(input_data[i])):
                        for xd in range(len(h1_out[0])):
                            input_weight_delta[d][xd] = (momentum * h1_err_unit[xd] * float(input_data[i][d])) + (
                                        l_rate * input_weight_delta[d][xd])

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
            if show_e_acc.get() == 1:
                print("%s.Epoch" % (str(j + 1)))
                print("Epoch True : " + str(e_true_count) + "   Epoch Rate : " + str(ep_rate) + "%  Accuracy :  " + str(
                    acc_rate) + "%")
            output_error.append(epoch_error)
            acc_array.append(acc_rate)

            if save_weight.get() == 1:
                # Ağırlıkların kaydedilmesi
                np.save("nn_weight/h1_weight", h1_weight)
                np.save("nn_weight/input_weight", input_weight)
                np.save("nn_weight/h1_bias", h1_bias)
                np.save("nn_weight/output_bias", output_bias)

        if show_e_weight.get() == 1:
            t_ar = np.array(all_epoch_weight)
            new_data_frame = pd.DataFrame(
                {
                    "W11" : t_ar.T[0],
                    "W12" : t_ar.T[1],
                    "W21" : t_ar.T[2],
                    "W22" : t_ar.T[3],
                    "W31" : t_ar.T[4],
                    "W32" : t_ar.T[5],
                    "W13" : t_ar.T[6],
                    "W23" : t_ar.T[7],
                    "WB1" : t_ar.T[8],
                    "WB2" : t_ar.T[9],
                    "WB3" : t_ar.T[10],
                },index=["%s.Epoch"%(str(i+1)) for i in range(len(all_epoch_weight))]
            )

            pd.set_option('expand_frame_repr', False)
            pd.set_option('display.max_row', None)
            print(new_data_frame)
            plt.plot(t_ar)
            plt.show()

        if show_i_weight.get() == 1:
            t_i_ar = np.array(all_it_weight)
            new_i_data_frame = pd.DataFrame(
                {
                    "W11": t_i_ar.T[0],
                    "W12": t_i_ar.T[1],
                    "W21": t_i_ar.T[2],
                    "W22": t_i_ar.T[3],
                    "W31": t_i_ar.T[4],
                    "W32": t_i_ar.T[5],
                    "W13": t_i_ar.T[6],
                    "W23": t_i_ar.T[7],
                    "WB1": t_i_ar.T[8],
                    "WB2": t_i_ar.T[9],
                    "WB3": t_i_ar.T[10],
                }, index=["%s.İterasyon" % (str(i + 1)) for i in range(len(all_it_weight))]
            )
            pd.set_option('expand_frame_repr', False)
            print(new_i_data_frame)
            plt.plot(t_i_ar)
            plt.show()


def testt():
    if len(select_arr) == 0:
        messagebox.showinfo("Hata", "Seçili örnek bulunamadı")
    else:
        print("asd")

if __name__ == "__main__":
    master = Tk()
    master.title("Makine Öğrenmesi Projesi")
    master.geometry("800x700")

    Label(master, text="X1", bg="lightgrey", height=2, width=10, ).grid(row=0, column=1, sticky=W, pady=(10, 0))
    Label(master, text="X2", bg="lightgrey", height=2, width=10, ).grid(row=0, column=2, sticky=W, pady=(10, 0))
    Label(master, text="X3", bg="lightgrey", height=2, width=10, ).grid(row=0, column=3, sticky=W, pady=(10, 0))
    Label(master, text="Y", bg="lightgrey", height=2, width=10, ).grid(row=0, column=4, sticky=W, pady=(10, 0))

    all_data = [
        [0, 0, 0, [0]],
        [0, 0, 1, [1]],
        [0, 1, 0, [0]],
        [0, 1, 1, [0]],
        [1, 0, 0, [0]],
        [1, 0, 1, [0]],
        [1, 1, 0, [1]],
        [1, 1, 1, [1]],
    ]


    select_arr = list()

    array = ['Örnek1', 'Örnek2', 'Örnek3', 'Örnek4', 'Örnek5', 'Örnek6', 'Örnek7', 'Örnek8', 'Örnek9']
    vars = []  # Array for saved values
    for y in range(8):
            vars.append(StringVar(value=0))
            Checkbutton(master,bg="lightgrey", height=2,width=10,text=array[y], command=lambda i=y: printSelection(i),onvalue='1',offvalue='0', variable=vars[-1]).grid(row=y + 1,sticky=W)
            # Label(master,bg="lightgrey", height=2,width=10, text="Örnek %s"%(y+1)).grid(row=y+1, sticky=W)
            Label(master,borderwidth=2, relief="groove",height=2,width=10, text=str(all_data[y][0])).grid(row=y+1,column=1,sticky=W)
            Label(master,borderwidth=2, relief="groove",height=2,width=10, text=str(all_data[y][1])).grid(row=y+1,column=2,sticky=W)
            Label(master,borderwidth=2, relief="groove",height=2,width=10, text=str(all_data[y][2])).grid(row=y+1,column=3,sticky=W)
            Label(master,borderwidth=2, relief="groove",height=2,width=10, text=str(all_data[y][3][0])).grid(row=y+1,column=4,sticky=W)

    lb_empty = Label(master, text="     ",padx=20,).grid(row=0,column=6,sticky=W)
    lb_epoch = Label(master, text="Epoch Sayısı").grid(row=0,column=7,sticky=W)
    e_count = IntVar(value=10)
    e_c = Entry(master, textvariable=e_count)
    e_c.grid(row=1,column=7,sticky=W)
    lb_learn = Label(master, text="Öğrenme Katsayısı").grid(row=2,column=7,sticky=W)
    l_rate = IntVar(value=0.1)
    l_r = Entry(master, textvariable=l_rate)
    l_r.grid(row=3,column=7,sticky=W)

    lb_mome = Label(master, text="Momentum Katsayısı").grid(row=4,column=7,sticky=W)
    mome = IntVar(value=0.1)
    m_m = Entry(master, textvariable=mome)
    m_m.grid(row=5,column=7,sticky=W)

    lb4 = Label(master, text="Hata Toleransı").grid(row=6,column=7,sticky=W)
    error_t = IntVar(value=0.05)
    e_t = Entry(master, textvariable=error_t)
    e_t.grid(row=7,column=7,sticky=W)

    save_weight = IntVar()
    s_w = Checkbutton(master, text="Ağırlıkları Kaydet", variable=save_weight)
    s_w.grid(row=8,column=7,sticky=W)

    use_weight = IntVar()
    u_w = Checkbutton(master, text="Kayıtlı Ağırlıkları Kullan", variable=use_weight)
    u_w.grid(row=9, column=7, sticky=W)

    show_i_weight = IntVar()
    s_i_w = Checkbutton(master, text="Her İterasyondaki Ağırlıkları Göster", variable=show_i_weight)
    s_i_w.grid(row=10, column=7, sticky=W)

    show_e_weight = IntVar()
    s_e_w = Checkbutton(master, text="Her Epochtaki Ağırlıkları Göster", variable=show_e_weight)
    s_e_w.grid(row=11, column=7, sticky=W)

    show_e_acc = IntVar()
    s_e_a = Checkbutton(master, text="Her Epochtaki Doğruluk Oranını Göster", variable=show_e_acc)
    s_e_a.grid(row=12, column=7, sticky=W)

    education_btn = Button(text="Ağı Eğit", command=nn_edu,padx=20,pady=5).grid(row=13,column=7,sticky=W)
    test_btn = Button(text="Ağı Test Et", command=testt,padx=20,pady=5).grid(row=14,column=7,sticky=W)

    mainloop()


