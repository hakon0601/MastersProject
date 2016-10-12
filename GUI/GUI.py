import tkinter as tk
from run_config_settings import *
import ttk

import data_loader as dl
from FeatureExtraction import short_time_fourier_transform, wavelet_transform, mel_frequency_cepstral_coefficients, spectral_density_estimation, no_feature_extraction
from NeuralNetwork import feed_forward_neural_network, convolutional_neural_network, recurrent_neural_network, radial_basis_function_neural_network
import main_program


class GUI(tk.Tk):

    feature_extraction_techniques = ["Short-time Fourier Transform",
                                     "Wavelet Transform",
                                     "Mel-frequency Cepstral Coefficients",
                                     "Spectral Density Estimation",
                                     "No Feature Extraction"
                                     ]
    neural_network_types = ["Standard Feed-forward Neural Network",
                            "Convolutional Neural Network",
                            "Recurrent Neural Network",
                            "Radial Basis Function Network"
                                     ]


    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        # self.center_window()

        self.build_parameter_menu()

    def build_parameter_menu(self):

        FEboxgroup = tk.Frame(self)

        tk.Label(FEboxgroup, text="Feature extraction technique").pack()
        self.FEbox_value = tk.StringVar()
        self.FEbox = self.combo(FEboxgroup, self.feature_extraction_techniques, self.FEbox_value)
        self.FEbox.pack()
        self.FEbox.bind("<<ComboboxSelected>>", self.newFEselection)
        FEboxgroup.pack(side=tk.LEFT)
        self.build_fe_options_menu()

        NNboxgroup = tk.Frame(self)
        tk.Label(self, text="Neural Network type").pack()
        self.NNbox_value = tk.StringVar()
        self.NNbox = self.combo(NNboxgroup, self.neural_network_types, self.NNbox_value)
        self.NNbox.pack()
        self.NNbox.bind("<<ComboboxSelected>>", self.newNNselection)
        NNboxgroup.pack(side=tk.RIGHT)

        button_group = tk.Frame(self)
        start_button = tk.Button(button_group, text="Start", width=20, command=self.start_program)
        start_button.pack()
        button_group.pack()
        self.build_nn_options_menu()

        #self.start_program()

    def start_program(self):
        data_loader = dl.DataLoader(TEST_PERCENTAGE, SAMPELING_RATE)
        FEtype = self.FEbox.get()
        if FEtype == self.feature_extraction_techniques[0]:
            feature_extractor = short_time_fourier_transform.STFT()
        elif FEtype == self.feature_extraction_techniques[1]:
            feature_extractor = wavelet_transform.WaveletTransform()
        elif FEtype == self.feature_extraction_techniques[2]:
            feature_extractor = mel_frequency_cepstral_coefficients.MFCC()
        elif FEtype == self.feature_extraction_techniques[3]:
            feature_extractor = spectral_density_estimation.SpectralDensityEstimation()
        elif FEtype == self.feature_extraction_techniques[4]:
            feature_extractor = no_feature_extraction.NoFE()

        NNtype = self.NNbox.get()
        if NNtype == self.neural_network_types[0]:
            neural_network = feed_forward_neural_network.FeedForwardNN()
        elif NNtype == self.neural_network_types[1]:
            neural_network = convolutional_neural_network.ConvolutionalNN()
        elif NNtype == self.neural_network_types[2]:
            neural_network = recurrent_neural_network.RecurrentNN()
        elif NNtype == self.neural_network_types[3]:
            neural_network = radial_basis_function_neural_network.RadialBasisFunctionNN()

        main_thread = main_program.MainProgram(feature_extractor, neural_network, data_loader=data_loader)

        


    def combo(self, frame, box_values, box_value):
        box = ttk.Combobox(frame, width=30, textvariable=box_value)
        box['values'] = box_values
        box.current(0)
        return box


    def newFEselection(self, event):
        value_of_combo = self.FEbox.get()
        self.build_fe_options_menu()
        print(value_of_combo)

    def newNNselection(self, event):
        value_of_combo = self.NNbox.get()
        self.build_nn_options_menu()
        print(value_of_combo)

    def build_fe_options_menu(self):
        fe_options_frame = tk.Frame()
        value_of_combo = self.FEbox.get()
        index = self.feature_extraction_techniques.index(value_of_combo)
        if index == 0:
            pass
            # Build options menu for stft
        elif index == 1:
            pass

    def build_nn_options_menu(self):
        self.nn_options_frame = tk.Frame(background='red')
        value_of_combo = self.NNbox.get()
        index = self.neural_network_types.index(value_of_combo)
        if index == 0:
            nr_of_layers_value = tk.StringVar()
            nr_of_layers_entry = tk.Entry(self.nn_options_frame, textvariable=nr_of_layers_value)
            nr_of_layers_entry.pack()
            print(nr_of_layers_entry.get())
            # Build options menu for stft
        elif index == 1:
            pass

        self.nn_options_frame.pack(side=tk.RIGHT)




    def center_window(self):
        ws = self.winfo_screenwidth()  # width of the screen
        hs = self.winfo_screenheight()  # height of the screen
        w = 1000
        h = 900
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)
        self.geometry('%dx%d+%d+%d' % (w, h, x, y))





if __name__ == "__main__":
    app = GUI()
    app.mainloop()
