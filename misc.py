"""
misc utils to make a deep NLP life easier.
"""
import collections
import os

import matplotlib.pyplot as plt
import numpy as np
import torch


class EasyPlot:
    """
    a collection of functions that plots data and outputs them as .png files.
    """
    @staticmethod
    def plot_heatmap(tensor, y_names, x_names, out_name: str, title: str, show=False):
        """
        credits to Iza.
        :param tensor: numpy ndarray or torch tensor of shape n×m, where n can equal m, but not nesessarily
        :param y_names: list of str : names of tickmarks on y axis; should be in a desired order; the names
        will go from top to bottom along the y axis
        :param x_names: list of str : names of tickmarks on x axis; should be in a desired order; the names
        will go from top to bottom along the x axis
        :param out_name: name of the saved plot
        :param title:
        :param show:
        :return:
        """

        # convert to a numpy array and round the values to 2 decimals to be able to fit the cells
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.numpy()
        Cnp = np.around(tensor, decimals=2)

        # set up the plot: size and tick marks
        fig, ax = plt.subplots(figsize=(14, 14))  # in inches, ~*2 to get cm
        im = ax.imshow(tensor)
        ax.set_xticks(np.arange(len(x_names)))
        ax.set_yticks(np.arange(len(y_names)))

        # tick labels
        ax.set_xticklabels(x_names)
        ax.set_yticklabels(y_names)
        # tick labels: position and rotation for columns
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # iteratively insert the cell values into the plot; in the middle and in white
        for i in range(len(y_names)):
            for j in range(len(x_names)):
                ax.text(j, i, Cnp[i, j], ha="center", va="center", color="w")

        # add the title to the plot
        ax.set_title(title)
        # add a colorbar
        plt.colorbar(im)
        fig.tight_layout()

        if show:
            plt.show()

        # save the plot as .png, but other formats are available (e.g. .svg or .jpg)
        plt.savefig(out_name)

    @staticmethod
    def plot_line_graph(x, y, x_name, y_name, out_name: str, title: str = '', mode=None, show=False):
        plt.plot(x, y, mode) if mode else plt.plot(x, y)
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        if len(title) > 0:
            plt.title(title)
        plt.savefig(f'{out_name}_.png', bbox_inches='tight')
        if show:
            plt.show()

    @staticmethod
    def plot_scatter_graph(x, y, x_name, y_name, out_name: str, title: str = '', color=None, size=None, show=False):
        plt.scatter(x=x, y=y, c=color, s=size)
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        if len(title) > 0:
            plt.title(title)
        plt.savefig(f'{out_name}_.png', bbox_inches='tight')
        if show:
            plt.show()


class PrintColors:
    """
        add some color to your terminal!
    """

    @staticmethod
    def prRed(prt): print("\033[91m {}\033[00m".format(prt))

    @staticmethod
    def prGreen(prt): print("\033[92m {}\033[00m".format(prt))

    @staticmethod
    def prYellow(prt): print("\033[93m {}\033[00m".format(prt))

    @staticmethod
    def prLightPurple(prt): print("\033[94m {}\033[00m".format(prt))

    @staticmethod
    def prPurple(prt): print("\033[95m {}\033[00m".format(prt))

    @staticmethod
    def prCyan(prt): print("\033[96m {}\033[00m".format(prt))

    @staticmethod
    def prLightGray(prt): print("\033[97m {}\033[00m".format(prt))

    @staticmethod
    def prBlack(prt): print("\033[98m {}\033[00m".format(prt))


class Struct(dict):
    """
    extend a dictionary so we can access its keys as attributes. improves quality of life with auto-completion.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            setattr(self, key, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        setattr(self, key, value)


class TextTokenizer:
    """
    a text tokenizer that would come handy with pytorch. Probably not needed for allenNLP though.
    """

    def __init__(self, text, vocabulary_size=-1):
        """
        :param text: a corpus as a list of tokens
        """
        assert text
        self._encoder = dict()
        if vocabulary_size == -1:
            token_counter = collections.Counter(text).most_common(len(text))
        else:
            token_counter = collections.Counter(text).most_common(vocabulary_size)
        for type_, count in token_counter:
            self._encoder[type_] = len(self._encoder)
        self._decoder = dict(zip(self._encoder.values(), self._encoder.keys()))

    def encode(self, text):
        if type(text) == str:
            return torch.tensor(self._encoder[text])
        else:
            return torch.tensor([self._encoder[text[i]] for i in range(len(text))])

    def decode(self, index):
        if (type(index) == torch.Tensor and len(index.size()) == 0) or \
                (type(index) == np.ndarray and len(index.shape) == 0) or type(index) == int:
            return self._decoder[index]
        else:
            return [self._decoder[index[i]] for i in range(len(index))]

    def append_type(self, new_type):
        if new_type in self._encoder:
            PrintColors.prRed('warning: type {} already exists.'.format(new_type))
        else:
            index = len(self._encoder)
            self._encoder[new_type] = index
            self._decoder[index] = new_type

    def append_type_s(self, new_type_s: list):
        for new_type in new_type_s:
            self.append_type(new_type)

    @property
    def vocabulary_size(self):
        return len(self._encoder)


""" generic tools """


def latex_table_of_csv(csv_file: str, output_file: str = 'latex_out', spliter='\t', alignment: str = ''):
    """
    generate latex codes that compiles a table from a csv file
    :param csv_file:
    :param output_file:
    :param spliter:
    :param alignment:
    :return:
    """
    splited_line_s = list()
    with open(csv_file, 'r') as insteam:
        for line in insteam:
            splited_line_s.append(line.split(spliter))
    _alignment = ''.join(['c']*len(splited_line_s[0])) if alignment == '' else alignment
    output = '\\begin{table}[htbp]\n\\begin{center}\n\\begin{tabular}' \
             + '{' + _alignment + '}' + '\\hline\n'
    for splited_line in splited_line_s:
        output += ' & '.join(splited_line) + '\\\\\n'
    output += '\\hline\n\\end{tabular}\n\\\\\\noindent *:say something sweetie\n' + \
              '\\caption{moin}\\label{yo}\n\\end{center}\n\\end{table}\n'
    output = output.replace('_', '\\_')
    with open(os.path.join('.', output_file), 'w') as outsteam:
        outsteam.write(output)
    print(output)
    return output


def fleiss_kappa(table):
    """
    fleiss kappa, see:
    http://en.wikipedia.org/wiki/Fleiss%27_kappa
    :param table: array-like, 2D
        t[i,j]: # rates assigning category j to subject i
    :return:
    """
    table = 1.0 * np.asarray(table)
    n_total = table.sum()
    n_rater = table.sum(1)
    n_rat = n_rater.max()

    # marginal frequency  of categories
    p_cat = table.sum(0) / n_total

    table2 = table * table
    p_rat = (table2.sum(1) - n_rat) / (n_rat * (n_rat - 1.))

    # annotation agreement
    p_mean = p_rat.mean()

    # random agreement
    p_e = (p_cat * p_cat).sum()

    kappa = (p_mean - p_e) / (1 - p_e)
    return kappa


def last_index_of(l, element):
    """ poor python, why do I need to implement this """
    if element not in l:
        return len(l)
    else:
        return len(l) - l[::-1].index(element) - 1


def pad_sequence(sequence, target_length: int, padding_token: str):
    return list(sequence) + [padding_token] * (target_length - len(sequence))


def preserve(f: float, n: int = 2):
    """ preserves n valid digits of float number f"""
    return float(('{:.' + str(n) + 'g}').format(f))


def to_categorical(y, num_classes):
    """
    projects a number to one-hot encoding
    """
    return np.eye(num_classes, dtype='uint8')[y]


if __name__ == '__main__':
    latex_table_of_csv('tmp')
