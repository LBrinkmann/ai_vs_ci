import matplotlib.pyplot as plt


def plot_screen(screen):
    plt.figure()
    plt.imshow(screen.squeeze(0).permute(1, 2, 0).numpy(),
            interpolation='none')
    plt.title('Example extracted screen')
    plt.show()


