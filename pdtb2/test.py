from pdtb2 import CorpusReader, Datum

iterator = CorpusReader('/home/btr/bpmn/pdtb2/pdtb2.csv').iter_data(display_progress=False)
for _ in range(17): next(iterator)

d = next(iterator)

d.arg1_words()
['that', '*T*-1', 'hung', 'over', 'parts', 'of', 'the', 'factory', ',']

d.arg1_words(lemmatize=True)
['that', '*T*-1', 'hang', 'over', 'part', 'of', 'the', 'factory', ',']

d.arg1_pos(wn_format=True)
[('that', 'wdt'), ('*T*-1', '-none-'), ('hung', 'v'), ('over', 'in'), \
('parts', 'n'), ('of', 'in'), ('the', 'dt'), ('factory', 'n'), (',', ',')]

d.arg1_pos(lemmatize=True)
[('that', 'wdt'), ('*T*-1', '-none-'), ('hang', 'v'), ('over', 'in'), \
('part', 'n'), ('of', 'in'), ('the', 'dt'), ('factory', 'n'), (',', ',')]

len(d.Arg1_Trees)
5

for t in d.Arg1_Trees:
	t.pprint()