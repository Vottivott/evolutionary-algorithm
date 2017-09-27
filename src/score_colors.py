import colorama
from colorama import Fore, Back, Style
colorama.init(autoreset=True)

color_lims = [(-60000, Fore.BLUE),
              (500, Fore.CYAN),
              (1000, Fore.GREEN),
              (1500, Fore.YELLOW),
              (2000, Fore.RED),
              (2500, Back.BLUE),
              (3000, Back.CYAN),
              (3500, Back.GREEN),
              (4000, Back.YELLOW),
              (4500, Back.RED),
              (60000, Back.MAGENTA)]

lims, col = zip(*color_lims)
neg_lims, col_rev = [-n for n in reversed(lims)], col
enemy_color_lims = zip(neg_lims, col_rev)

# print Fore.RED + "HEJ"

# print color_lims
# print enemy_color_lims

def get_color_from_score(score, is_enemy):
    if is_enemy:
        lm = enemy_color_lims
    else:
        lm = color_lims
    i = 0
    while i < len(lm) and score > lm[i][0]:
        i+=1
    if i >= len(lm):
        i = len(lm)-1
    return lm[i][1]

def colored_score(score, is_enemy):
    return get_color_from_score(score, is_enemy) + str(score)

# for i in range(-50,6500,50):
#     print colored_score(-i, 1)
