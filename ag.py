import datetime
import random

random.seed(int(str(hash(datetime.date.today()))))
the_board = ['Alexander', 'Arne', 'Fangzhou', 'Jonas', 'Lucia', 'Meaghan', 'Yuekun', 'Iza', 'Debanjali']
random.shuffle(the_board)
for participant in (['==================== WELCOME TO THE GLORIOUS MINI MEETING ====================']
                    + the_board + ['==============================================================================']):
    print("\033[98m {}\033[00m" .format(participant))
