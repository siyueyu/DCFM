import os


class Config():
    def __init__(self) -> None:

        # Performance of GCoNet
        self.val_measures = {
            'Emax': {'CoCA': 0.783, 'CoSOD3k': 0.874, 'CoSal2015': 0.892},
            'Smeasure': {'CoCA': 0.710, 'CoSOD3k': 0.810, 'CoSal2015': 0.838},
            'Fmax': {'CoCA': 0.598, 'CoSOD3k': 0.805, 'CoSal2015': 0.856},
        }

        # others


        self.validation = True