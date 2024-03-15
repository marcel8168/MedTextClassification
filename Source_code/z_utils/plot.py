from matplotlib import pyplot as plt


def plot(data_list: list, data_label_list: list[str], title, ylabel, xlabel, vlines: list = [], ylim=None, model_name: str = None):
    for i, data in enumerate(data_list):
        plt.plot(data, label=data_label_list[i])
    for i, vline in enumerate(vlines):
        plt.axvline(x=vline, color="tab:orange", linestyle="dotted",
                    label="End of epoch" if i == 0 else None)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if ylim:
        plt.ylim(ylim)
    if len(data_list) > 1 or len(vlines) > 0:
        plt.legend()
    model_name = model_name[model_name.find('/')+1:] if model_name else ""
    plt.savefig(
        f"{title}_{model_name}.pdf", format="pdf", bbox_inches="tight")
    plt.show()
    plt.close()
