from itertools import product
import csv

'''
phone_feature_map = {
    'M': ('blb', 'nas'),
    'P': ('vls', 'blb', 'stp'),
    'B': ('vcd', 'blb', 'stp'),
    'F': ('vls', 'lbd', 'frc'),
    'V': ('vcd', 'lbd', 'frc'),
    'TH': ('vls', 'dnt', 'frc'),
    'DH': ('vcd', 'dnt', 'frc'),
    'N': ('alv', 'nas'),
    'T': ('vls', 'alv', 'stp'),
    'D': ('vcd', 'alv', 'stp'),
    'S': ('vls', 'alv', 'frc'),
    'Z': ('vcd', 'alv', 'frc'),
    'R': ('alv', 'apr'),
    'L': ('alv', 'lat'),
    'SH': ('vls', 'pla', 'frc'),
    'ZH': ('vcd', 'pla', 'frc'),
    'Y': ('pal', 'apr'),
    'NG': ('vel', 'nas'),
    'K': ('vls', 'vel', 'stp'),
    'G': ('vcd', 'vel', 'stp'),
    'W': ('lbv', 'apr'),
    'HH': ('glt', 'apr'),
    'CH': ('vls', 'alv', 'stp', 'frc'),
    'JH': ('vcd', 'alv', 'stp', 'frc'),
    'AO': ('lmd', 'bck', 'rnd', 'vwl'),
    'AA': ('low', 'bck', 'unr', 'vwl'),
    'IY': ('hgh', 'fnt', 'unr', 'vwl'),
    'UW': ('hgh', 'bck', 'rnd', 'vwl'),
    'EH': ('lmd', 'fnt', 'unr', 'vwl'),
    'IH': ('smh', 'fnt', 'unr', 'vwl'),
    'UH': ('smh', 'bck', 'rnd', 'vwl'),
    'AH': ('mid', 'cnt', 'unr', 'vwl'),
    'AE': ('low', 'fnt', 'unr', 'vwl'),
    'EY': ('lmd', 'smh', 'fnt', 'unr', 'vwl'),
    'AY': ('low', 'smh', 'fnt', 'cnt', 'unr', 'vwl'),
    'OW': ('umd', 'smh', 'bck', 'rnd', 'vwl'),
    'AW': ('low', 'smh', 'bck', 'cnt', 'unr', 'rnd', 'vwl'),
    'OY': ('lmd', 'smh', 'bck', 'fnt', 'rnd', 'unr', 'vwl'),
    'ER': ('umd', 'cnt', 'rzd', 'vwl'),
    '^': ('beg',),
    '$': ('end',)
}
'''

# generate mappings between ipa marks and sets of phoneme features
phone_feature_map = {}
with open('ipa2feature.csv', encoding='utf-8-sig') as f:
    f_csv = csv.reader(f)
    for row in f_csv:
        #row[0] = row[0].replace("\ufeff", "")

        if row[0] != '':
            phone_feature_map[row[0]] = row[1].split()

phone_feature_map['^'] = ['beg',]
phone_feature_map['$'] = ['end',]

def phone_to_features(ph):
    if ph[-1] in '012':
        ph = ph[:-1]
    if ph[-1] == 'ʰ':
        return phone_feature_map[ph[:-1]] + ['asp']
    elif ph[-1] == 'ʲ':
        return phone_feature_map[ph[:-1]] + ['pzd']
    elif ph[-1] == '̃':
        return phone_feature_map[ph[:-1]] + ['nzd']
    elif ph[-1] == "ᵝ":
        return phone_feature_map[ph[:-1]] + ['cmp']
    else:
        return phone_feature_map[ph]

def feature_bigrams(phones_list, include_reverse = False):

    # find n-grams of each successive pair
    grams = list()
    phones_list = ["^"] + phones_list + ["$"]

    for ph0, ph1 in zip(phones_list[:-1], phones_list[1:]):
        for item in product(*[phone_to_features(ph0), phone_to_features(ph1)]):
            grams.append('-'.join(item))

    # backwards too
    if include_reverse:
        phones_list = list(reversed(phones_list))
        for ph0, ph1 in zip(phones_list[:-1], phones_list[1:]):
            for item in \
                    product(*[phone_to_features(ph0), phone_to_features(ph1)]):
                grams.append('-'.join(item))

    return grams

if __name__ == '__main__':
    import doctest
    doctest.testmod()

