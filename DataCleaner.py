import numpy as np
import re
import pandas as pd

#Remove more than one .
r1='\.{2,}|\d+\.*|ಬಿಏಖಿಃS\n|ಓoಣ ಣo be ಡಿeಠಿubಟisheಜ\n'
#Kannada character range
r4='[ ಂ-೯\s\.]'
#ಅ to ಹ
r5='[ ಂ-ಹ]'
#Sentence break
r6 = '\.'

r7='[ಅ-ಹ]'



class DataCleaner:

    def __init__(self, df):
        self.df=df
    def filter_chrs(self, line):
        filtered_line=re.sub(r1,"",line)
        filtered_line=re.sub('\n'," ",filtered_line)
        filtered_line= ''.join(re.findall(r4,filtered_line))
        filtered_line=re.sub('\s{2,}',' ',filtered_line)
        return self.trim_1_chr(filtered_line.strip())


    def clean_gadya(self):
        X = self.df['message']
        new_lines=[]
        for line in X:
            line = self.filter_chrs(line)
            new_line = line.strip()
            cnt, _ = self.count_chrs(new_line)
            if cnt > 2:
                new_lines.append(new_line)
        self.df['message']=new_lines
        return self.df

    def count_chrs(self, word):
        chrs = len(re.findall(r7, word))
        ottakshara = word.count('್')
        return chrs-ottakshara, ottakshara


    def trim_1_chr(self, sent):
        words=[]
        for w in sent.split(' '):
            if len(w) !=1:
                words.append(w)
        return ' '.join(words)
