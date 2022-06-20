import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    y_pred_mat = np.random.uniform(size=(3,3))
    print(y_pred_mat)

    plt.imshow(y_pred_mat, cmap='Reds')
    # 每个方格上标记数值
    for i in range(3):
        for j in range(3):
            text = plt.text(j, i, '{:.3f}'.format(y_pred_mat[i, j]),
                            ha="center", va="center", color="black", fontsize=8)
    plt.show()
