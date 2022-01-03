import numpy as np 
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

class BananaDataset():
    """
        Banana-shaped dataset generation and partitioning.
        
    """
    def generate(**kwargs):
        # Banana-shaped dataset generation
        number = kwargs['number']
        display = kwargs['display']
        
        # parameters for banana-shaped dataset
        sizeBanana = 3
        varBanana = 1.2
        param_1 = 0.02
        param_2 = 0.02
        param_3 = 0.98
        param_4 = -0.8 # x-axsis shift
        # generate 
        class_p = param_1 * np.pi+np.random.rand(number, 1) * param_3 * np.pi
        data_p = np.append(sizeBanana * np.sin(class_p), sizeBanana * np.cos(class_p), axis=1)
        data_p = data_p + np.random.rand(number, 2) * varBanana
        data_p[:, 0] = data_p[:, 0] - sizeBanana * 0.5
        label_p = np.ones((number, 1), dtype=np.int64)
        
        class_n = param_2 * np.pi - np.random.rand(number, 1) * param_3 * np.pi
        data_n = np.append(sizeBanana * np.sin(class_n), sizeBanana * np.cos(class_n), axis=1)
        data_n = data_n + np.random.rand(number, 2)*varBanana
        data_n = data_n + np.ones((number, 1)) * [sizeBanana * param_4, sizeBanana * param_4]
        data_n[:, 0] = data_n[:, 0] + sizeBanana * 0.5
        label_n = -np.ones((number, 1), dtype=np.int64)
        
        # banana-shaped dataset
        data = np.append(data_p, data_n, axis=0)
        label = np.append(label_p, label_n, axis=0)
        
        if display == 'on':
            pIndex = label == 1
            nIndex = label == -1
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(1, 1, 1)
            ax.scatter(data[pIndex[:, 0], 0], data[pIndex[:, 0], 1],
                    facecolor='C0', marker='o', s=100, linewidths=2,
                    edgecolor='black', zorder=2)
            
            ax.scatter(data[nIndex[:, 0], 0], data[nIndex[:, 0], 1],
                    facecolor='C3', marker='o', s = 100, linewidths=2,
                    edgecolor='black', zorder=2)
            
            ax.set_xlim([-6, 5])
            ax.set_ylim([-7, 7])
        
        return data, label
    
    def split(data, label, **kwargs):
        # Banana-shaped dataset partitioning.
        
        ratio = kwargs['ratio']
        X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=ratio,
                                                             random_state=None, shuffle=True, stratify=label)
        pIndex = y_train == 1
        nIndex = y_train == -1
        X_train = np.append(X_train[pIndex[:, 0], :], X_train[nIndex[:, 0], :], axis=0)
        y_train = np.append(y_train[pIndex[:, 0], :], y_train[nIndex[:, 0], :], axis=0)

        pIndex = y_test == 1
        nIndex = y_test == -1
        X_test = np.append(X_test[pIndex[:, 0], :], X_test[nIndex[:, 0], :], axis=0)
        y_test = np.append(y_test[pIndex[:, 0], :], y_test[nIndex[:, 0], :], axis=0)
        
        return X_train, X_test ,y_train, y_test