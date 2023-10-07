from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format("cc.en.300.vec", binary=False, limit=600000)

keys = list(model.key_to_index.keys())
vowels = ['a', 'e', 'i', 'o', 'u', 'y']
keys = [w for w in keys if not any(c.isdigit() or c in ['.', ',', '\''] or c.isupper() for c in w) and any(c in vowels for c in w) and not w[0] == '-']
keys.append('I')
keys.append('T-shirt')
need_to_remove = ['i', 'und', 'ofthe', 'the', 'en', 'url', 'em', 'google', 'ii', 'href', 'cgi-bin', 'iv', 'tbe', 'urls', 'img', 'bgcolor', 'xo', 'vi', 'ti', 'ab', 'ce', 'ap', 'uid', 'af', 'lib', 'qui', 'fr-en', 'ci', 'jpeg', 'permalinkembedsaveparentgive', 'permalinkembedsavegive', 'aux', 'ai', 'rsquo', 'tis', 'ir', 'epub', 'cu', '<a', 'tel', 'seq', 'digi', 'it-', 'tions', 'zum', 'tothe', 'crit', 'ea', 'ji', 'vous', 'bei', 'itunes', 'steve', 'foo', 'ser', 'ge', 'tne', 'ei', 'ohio', 'idk', 'ra', 'io', 'ag', 'fra', 'sol', 'cdot', 'apk', 'ja', 'ae', 'sam', 'thi', 'phi', 'mais', 'ro', 'gifs', 'vu', 'oc', 'isbn', 'adj', 'vii', 'api', 'lede', 'al', 'la', 've', 'id', 'com', 'ing', 'frac', 'op', 'av', 'os', 'usb', 'psi', 'rec', 'rom', 'copyvio', 'cpu', 'usr', 'gif', 'atm', 'diy', 'ic', 'dpi', 'mba', 'ans', 'va', 'abt', 'faves', 'wii', 'permalinksavecontextfull', 'uns', 'ilk', 'milfhunter', 'mysql', 'enwiki', 'ui', 'iffy', 'amd', 'pid', 'xi', 'att', 'ich', 'lakhs', 'adb', 'o', 'ups', 'iphone', 'javascript', 'linux', 'si', 'ar', 'wordpress', 'ga', 'tus', 'ev', 'ipad', 'ipod', 'instagram', 'bios', 'xoxo', 'wp-content', 'muy', 'ep', 'ka', 'xbox', 'exp', 'fa', 'lor', 'utm', 'postsre', 'tor', 'y', 'cum', 'mater', 'por', 'po', 'col', 'tot', 'oct', 'ved', 'wir', 'jour', 'pe', 'twink', 'ob', 'perl', 'one-', 'lhe', 'mathbb', 'wav', 'ios', 'fe', 'ao', 'gi', 'yu', 'lu', 'ubuntu', 'über', 're', 'de', 'et', 'e', 'u', 'ad', 'min', 'ads', 'im', 'pro', 'del', 'di', 'amp', 'sec', 'ya', 'mi', 'du', 'der', 'da', 'ed', 'ex', 'ha', 'oz', 'le', 'ie', 'est', 'un', 'uk', 'se', 'ur', 'ca', 'doc', 'ol', 'er', 'cam', 'na', 'el', 'tee', 'lo', 'es', 'til', 't-shirt', 'rep', 'yd', 'yr', 'yrs', 'um', 'co', 'au', 'iii', 'eg', 'les', 'mho', 'ive', 'ma', 'hist', 'il', 'ot', 'mo', 'ho', 'sa', 'te', 'def', 'sic', 'li', 'ne', 'usa', 'ac', 'clubnat', 'gen', 'una', 'wa', 'org', 'hes', 'ou', 'ip', 'pas', 'tha', 'res', 'pa', 'fo', 'og', 'pi', 'ni', 'bo', 'fi', 'homer', 'ben', 'iso', 'ta', 'zu', 'vids', 'su', 'das', 'eng', 'dis', 'wi', 'ru', 'ter', 'ul', 'meth', 'wo', 'tu', 'mu', 'tracy', 'je', 'lon', 'dem', 'joe', 'gov', 'iu', 'ly', 'bi', 'oe', 'ia', 'ki', 'om', 'oi', 'ba', 'esl', 'aa', 'ugg', 'git', 'yer', 'ugh', 'louis', 'een', 'hon', 'eu', 'seo', 'ops', 'pis', 'aud', 'dir', 'od', 'esp', 'mol', 'deg', 'hee', 'yds', 'emo', 'ami', 'ein', 'bona', 'rad', 'talo', 'mar', 'hus', 'omg', 'paul', 'fic', 'ke', 'che', 'nelle', 'bu', 'ty', 'ado', 'ko', 'fu', 'ny', 'eb', 'het', 'fiat', 'ut', 'lee', 'ang', 'hai', 'chris', 'ala', 'dec', 'pix', 'som', 'dal', 'jus', 'ave', 'ons', 'avg', 'poi', 'pong', 'hoo', 'ana', 'nu', 'luv', 'tau', 'bono', 'mai', 'rel', 'aus', 'ted', 'nov', 'nav', 'jan', 'loo', 'als', 'jay', 'ohm', 'dia', 'fer', 'mis', 'pom', 'mer', 'wha', 'vis', 'tim', 'apartemen', 'nat', 'twat', 'pov', 'mas', 'fam', 'ow', 'ri', 'alia', 'nyc', 'cha', 'keg', 'chow', 'ent', 'feb', 'umm', 'qu', 'vat', 'mani', 'nad', 'bla', 'dave', 'kim', 'ber', 're-', 'ble', 'har', 'dun', 'sid', 'gee', 'gras', 'hy', 'eff', 'sont', 'tak', 'cor', 'ry', 'sag', 'sen', 'wen', 'nel', 'lagu', 'aw', 'mos', 'pug', 'ef', 'rho', 'kan']
print(len(need_to_remove))
for w in need_to_remove:
    try:
        keys.remove(w)
    except ValueError:
        print(w)
print(keys[26900:27000])
del_lst = []
all_lst = []
with open('necessary.txt', encoding='utf-8') as f:
    content = f.readlines()
    for row in content:
        label = row.rstrip()
        all_lst.append(label)
        if label not in keys:
            # print(label)
            del_lst.append(label)
print(len(del_lst), del_lst)
rem_lst = [i for i in all_lst if i not in del_lst]
for word in rem_lst:
    print(word)

with open('things3.txt', 'w', newline='', encoding='utf-8') as f:
    for label in keys[:26565]:
        f.write(label + "_\n")
    i = 0
    for label in rem_lst:
        if label not in keys[:26565]:
            f.write(label + "_\n")
            i += 1
    print(i)
with open('model.txt', 'w', newline='', encoding='utf-8') as f:
    for label in keys[:26565]:
        f.write(label + '_ ' + ' '.join(["{:.4f}".format(val) for val in model[label]]) + "\n")
    for label in rem_lst:
        if label not in keys[:26565]:
            f.write(label + '_ ' + ' '.join(["{:.4f}".format(val) for val in model[label]]) + "\n")
