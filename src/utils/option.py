import csv

from config import SETTINGS
from utils.constants import Suggestion


class Option(object):

    def __init__(self) -> None:
        self.option = self.get_option()
        self.str = ''.join(self.option)
    
    def __str__(self) -> str:
        return self.str
        
    def get_suggestion(self, option):
        return [Suggestion.BLOCK, Suggestion.PASS, Suggestion.REVIEW][self.option.index(option)]
    
    @staticmethod 
    def get_option():
        with open(SETTINGS.option_file, "r") as option_file:
            option = list(csv.reader(option_file))
            
        option = list(map(lambda x: x[0], option))
        return option