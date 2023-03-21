import os


class LossWriter():
    def __init__(self, save_dir):
        """
        initialize，loss_writer = LossWriter("xxx/")
        :param save_dir: path to sve the loss with txt format
        """
        self.save_dir = save_dir

    def add(self, loss_name, loss, i):
        """
        writing the loss to txt
        :param loss_name: name of loss txt
        :param loss: loss value
        :param i: current iterations
        :return: None
        """
        with open(os.path.join(self.save_dir, loss_name + ".txt"), mode="a") as f:
            term = str(i) + " " + str(loss) + "\n"
            f.write(term)
            f.close()