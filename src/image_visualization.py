import matplotlib.pyplot as plt


# Show attention
def plot_attention(img, result, attention_plot, image_dir):
    # img = img.numpy().transpose((1, 2, 0))
    temp_image = img

    fig = plt.figure(figsize=(15, 15))

    len_result = len(result)
    for l in range(len_result):
        temp_att = attention_plot[l][1:].reshape(14, 14)
        # temp_att = np.resize(attention_plot[l].detach().numpy(),(98,98))
        ax = fig.add_subplot(len_result // 2, len_result // 2, l + 1)
        ax.set_title(result[l], fontsize=18)
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, alpha=0.6, cmap="jet", extent=img.get_extent())

    plt.tight_layout()
    plt.savefig(image_dir)
