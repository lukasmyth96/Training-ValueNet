import ntpath


class DataItem:
    def __init__(self, filepath, class_name):
        self._filepath = filepath
        self.filename = ntpath.basename(filepath)  # npath is compatible with any os
        self._data = None
        self._feature_vector = None
        self._class_name = class_name

        self.tv_point_estimates = dict()  # maps (episode, iteration) --> point_estimate
        self.estimated_tv = None  # estimated training-value from MC estimation
        self.predicted_tv = None  # predicted training-value from Training-ValueNet

    @property
    def filepath(self):
        return self._filepath

    @property
    def data(self):
        return self._data

    @property
    def class_name(self):
        return self._class_name