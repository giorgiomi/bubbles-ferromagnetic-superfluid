import numpy as np

def importParameters(selected_flag):
    # Importing data from raw
    f = [r"data/raw/08/seq_89.hdf", # database of a certain day
        r"data/raw/08/seq_2.hdf",
        r"data/raw/08/seq_8.hdf",
        r"data/raw/11_14/seq_1.hdf",
        r"data/raw/11_11/seq_1.hdf",
        r"data/raw/11_25/seq_1.hdf",
        r"data/raw/11_29/seq_1.hdf",
        r"data/raw/11_30/seq_1.hdf",
        r"data/raw/01_13/seq_1.hdf",
        r"data/raw/01_16/seq_1.hdf",
        r"data/raw/01_17/seq_1.hdf",
        r"data/raw/01_19/seq_1.hdf",
        r"data/raw/01_24/seq_1.hdf",
        r"data/raw/07_18/seq_1.hdf",
        r"data/raw/07_19/seq_1.hdf",
        r"data/raw/07_22/seq_1.hdf",
        r"data/raw/07_23/seq_1.hdf",
        r"data/raw/07_25/seq_1.hdf",
        r"data/raw/07_26/seq_1.hdf",
        r"data/raw/07_29/seq_1.hdf",
        r"data/raw/08_02/seq_1.hdf",
        r"data/raw/08_06/seq_1.hdf",
        r"data/raw/08_07/seq_1.hdf"]
       

    # Choosing sequences
    seqs = [[[5],[6],[7],[8]], #sequenze utile del giorno tot
        [[8],[9],[10]],
        [[10],[14]],
        [[9],[18,21,24,25]],
        [[24,25,26],[28,29,30],[31],[32,33],[34],[35],[37,38]],
        [[12,14,20,24],[23,25],[21,26],[22,27],[10]],
        [[5],[6],[8],[9,10],[33,39,48],[45]],
        [[9],[10,18,19,20],[11],[13]],
        [[12,13,17,18],[14,15,19,25,27,26]],
        [[12],[13,14]],
        [[33,34]],
        [[15],[16],[17,18]],
        [[22,23,29,30],[24,25]],
        [[23,24,25,26],[31,32]],
        [[39,40,41,44,45,46,47,49],[59, 60, 61, 62],[68, 69],[68, 69]],
        [[37,39],[37,39],[28]],
        [[40, 41, 45, 46, 47, 48, 49, 50, 51],[40, 41, 45, 46, 47, 48, 49, 50, 51]],
        [[7, 8, 10, 12],[7, 8, 10, 12], [14, 15, 18], [14, 15, 18]],
        [[45, 46, 47], [45, 46, 47]],
        [[26, 27, 28, 29], [26, 27, 28, 29]],
        [[36, 37, 38, 39], [46, 47]],
        [[22],[27, 28], [42, 43]],
        [[27],[28,29,30,34]]]

    if selected_flag:
        # Choosing better sequences
        selected_days = [0, 1, 2, 3, 6, 7, 8, 9, 10, 12]
        selected_seqs = [[0, 1, 2, 3],
                        [0, 1, 2],
                        [0, 1],
                        [0, 1],
                        [], 
                        [],
                        [0, 4],
                        [0, 1],
                        [0, 1],
                        [0, 1],
                        [0],
                        [],
                        [1]]
    else:
        # selecting all seqs
        selected_days = range(len(seqs))
        selected_seqs = [range(len(seq)) for seq in seqs]

    # A lot of tuning parameters
    Conv=2700

    Ref2=0.326
    Ref2b=0.335
    Ref3=0.145
    Ref4=0.345
    Ref5=0.39
    Ref6=0.31
    Ref7=0.335
    Ref8=0.207
    Ref9=0.22
            

    HystEnd=[[0.2,0.2,0.2,0.2], #400
            [0.225,0.225,0.225], #400
            [0.245,0.245], #400
            [Ref2,Ref2], #600
            [Ref2b,Ref2b,Ref2b,Ref2b,Ref2b,Ref2b,Ref2b], #600
            [Ref3,Ref3,Ref3,Ref3,Ref3], #200
            [Ref4,Ref4,Ref4,Ref4,0.405,0.405], #800
            [Ref5,Ref5,Ref5,Ref5], #800
            [Ref6,Ref6], #600 2023
            [Ref6,Ref6],
            [Ref7], #600 2023
            [Ref8,Ref8,Ref8],
            [Ref9,Ref9]] #300 2023

    Offset=[[0,0,0,0], #400
            [0.025,0.025,0.025], #400
            [0.045,0.045], #400
            [0.03,0.03], #600
            [0.03,0.03,0.03,0.03,0.03,0.03,0.03], #600
            [0,0,0,0,0], #200
            [0,0,0.0,0.0,0.06,0.06], #800
            [0.05,0.05,0.05,0.05], #800
            [0,0],#600 2023
            [0,0],#600 2023
            [0.025],#600 2023
            [0,0,0],
            [0.013,0.013]]

    Srs_V=[[0.205,0.210,0.235,0.225], #400
        [0.275,0.265,0.265], #400
        [0.250,0.265], #400
        [0.34,0.35], #600
        [0.381,0.34,0.35,0.36,0.37,0.38,0.39], #600
        [0.15,0.19,0.17,0.18,0.2], #200
        [0.36,0.37,0.38,0.39,0.42,0.45], #800
        [0.40,0.42,0.44,0.45], #800
        [0.35,0.36], #600 2023
        [0.33,0.32],
        [0.40], #600 2023
        [0.22,0.21,0.25],
        [0.25,0.24]]
        
    Om1=400
    Om2=600
    Om3=200
    Om4=800
    Om5=300
            
    Omega=[[Om1,Om1,Om1,Om1],
       [Om1,Om1,Om1],
       [Om1,Om1],
       [Om2,Om2],
       [Om2,Om2,Om2,Om2,Om2,Om2,Om2],
       [Om3,Om3,Om3,Om3,Om3],
       [Om4,Om4,Om4,Om4,Om4,Om4],
       [Om4,Om4,Om4,Om4],
       [Om2,Om2],
       [Om2,Om2],
       [Om2],
       [Om5,Om5,Om5],
       [Om5,Om5],
       [Om5,Om5],
       [Om5,Om5,Om5,Om5],
       [Om5,Om5,Om5],
       [Om5,Om5],
       [Om5,Om5,Om5,Om5],
       [Om5,Om5],
       [Om5,Om5],
       [Om5,Om5],
       [Om5,Om5,Om5],
       [Om5,Om5]]
            
    kn1=1150/Conv
    kn2=kn1
    kn3=kn1
    kn4=kn1
    kn5=kn1
    kn6=kn1
    kn7=kn1
    kn8=kn1

    knT=[[kn1,kn1,kn1,kn1], #400
        [kn1,kn1,kn1], #400
        [kn1,kn1], #400
        [kn2,kn2],#600
        [kn2,kn2,kn2,kn2,kn2,kn2,kn2],#600
        [kn3,kn3,kn3,kn3,kn3], #200
        [kn4,kn4,kn4,kn4,kn4,kn4], #800
        [kn5,kn5,kn5,kn5], #800
        [kn6,kn6],#600 2023
        [kn6,kn6],
        [kn7], #600 2023
        [kn8,kn8,kn8],
        [kn8,kn8]]

    detuningCal=np.array(Conv*((np.array(sum(knT,[])))+(np.array(sum(Offset,[])))-np.array(sum(Srs_V,[])))/1)
    detuningEnd=np.array(Conv*(-(np.array(sum(HystEnd,[])))+np.array(sum(Srs_V,[])))/1)

    detuning=[detuningCal[0:4],detuningCal[4:7],
            detuningCal[7:9],detuningCal[9:11],
            detuningCal[11:18],
            np.array(detuningCal[18:23]),
            np.array(detuningCal[23:29]),
            np.array(detuningCal[29:33]),
            np.array(detuningCal[33:35]),
            np.array(detuningCal[35:37]),
            np.array(detuningCal[37:38]),
            np.array(detuningCal[38:41]),
            np.array(detuningCal[41:43]),
            np.array(detuningCal[43:44])]

    # detuningCal2 = detuningCal
    # detuningkn = detuningCal/sum(knT,[])/Conv
    # detuningOm = detuningCal/sum(Omega,[])
    # detuningOmEnd = detuningEnd/sum(Omega,[])

    return f, seqs, Omega, knT, detuning, selected_days, selected_seqs