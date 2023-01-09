import os, math
from collections import Counter

class NaiveBayesClassifier: #Class(sınıfın tanımlanması)

    """
    Bir kelime torbası Naive Bayes sınıflandırıcısı için kod .
    """
    
    def __init__(self, train_dir='haiti/train', REMOVE_STOPWORDS=False): #Değişkenin objeye tanımlandığı yer Örnek Kullanım: clf = NaiveBayesClassifier(train_dir = 'haiti/train')
        self.REMOVE_STOPWORDS = REMOVE_STOPWORDS
        self.stopwords = set([l.strip() for l in open('english.stop')])
        self.classes = ['irrelevant', 'relevant']
        self.train_data = {'irrelevant': 'haiti/train', 'relevant': 'haiti/train'}
        self.vocabulary = set([])
        self.logprior = {}
        self.loglikelihood = {} # keys should be tuples in the form (w, c)

        
    def tokenize(self, document): # Verilen döküman içerisindeki features(belirtici özellik)leri çıkarır.
        """
        Belirli bir belge için kelime çantası özelliklerini çıkarır.
        """
        if self.REMOVE_STOPWORDS:
            document = ' '.join([w for w in document.split() if w not in self.stopwords])
        return set(document.split())

    def train(self): # bulunan belirtlici özellikler ile eğitimi başlatır.
        """
        Naive Bayes sınıflandırıcısını eğitim verileri üzerinde eğitin.
        """
        # Eğitim verilerinden bir sözcük dağarcığı oluşturun
        # print(self.train_data)
        for c in self.classes:
            for f in os.listdir(self.train_data[c]):
                with open(self.train_data[c]+'/'+f, 'r') as doc:
                    text = doc.read()
                    if self.REMOVE_STOPWORDS:
                        text = ' '.join([w for w in text.split() if w not in self.stopwords])
                    self.vocabulary.update(text.split())

        # Her sınıf için log önceliği olasılıklarını hesaplayın
        num_docs = sum([len(os.listdir(self.train_data[c])) for c in self.classes])
        for c in self.classes:
            self.logprior[c] = math.log(len(os.listdir(self.train_data[c])) / num_docs)

        # Her kelime ve sınıf için günlük olasılığını hesaplayın
        self.loglikelihood = {}
        for c in self.classes:
            counter = Counter()
            for f in os.listdir(self.train_data[c]):
                with open(self.train_data[c]+'/'+f, 'r') as doc:
                    text = doc.read()
                    if self.REMOVE_STOPWORDS:
                        text = ' '.join([w for w in text.split() if w not in self.stopwords])
                    counter.update(text.split())
            for w in self.vocabulary:
                self.loglikelihood[w, c] = math.log((counter[w] + 1) / (sum(counter.values()) + len(self.vocabulary)))

    def score(self, doc, c): # Belirlenen döküman içindeki sınıfa göre örn: IRRELEVANT skor hesaplar
        """
        Belirli bir sınıf için bir belgenin puanını hesaplayınmı.
        """
        score = self.logprior[c]
        for w in doc.split():
            if w in self.vocabulary:
                score += self.loglikelihood[w, c]
        return score

    def predict(self, doc): # Eğitim sonrası oluşan modelin tahmin fonksiyonudur.
        """
        Bir belge için sınıfı tahmin edin.
        """
        scores = {c: self.score(doc, c) for c in self.classes}
        return max(scores, key=scores.get)

    def evaluate(self, test_dir='haiti/test', target='relevant'): # Oluşturulan modelin doğruluk skorlarını hesaplar(f1 score,recall, accuracy)
        """
        erilen test verileri üzerinde sınıflandırıcının performansını değerlendirin.
        """
        test_data = {'irrelevant': 'haiti/dev', 'relevant': 'haiti/dev'}
        if not target in test_data:
            print('Hata: test verilerinde hedef sınıf yok.')
            return
        outcomes = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
        for c in self.classes:
            for f in os.listdir(test_data[c]):
                with open(os.path.join(test_data[c]+'/'+f),'r') as doc:
                    text = doc.read()
                    if self.predict(text) == target:
                        if c == target:
                            outcomes['TP'] += 1
                        else:
                            outcomes['FP'] += 1
                    else:
                        if c == target:
                            outcomes['FN'] += 1
                        else:
                            outcomes['TN'] += 1
        try:
            precision = outcomes['TP'] / (outcomes['TP'] + outcomes['FP'])
        except:
            precision=1.00
        recall = outcomes['TP'] / (outcomes['TP'] + outcomes['FN'])
        f1_score = 2 * precision * recall / (precision + recall)
        return (precision, recall, f1_score)


    def print_top_features(self, k=5): # Belirtici özellikleri ekrana yaazdırır
        results = {c: {} for c in self.classes}
        for w in self.vocabulary:
            for c in self.classes:
                ratio = math.exp( self.loglikelihood[w, c] - min(self.loglikelihood[w, other_c] for other_c in self.classes if other_c != c) )
                results[c][w] = ratio

        for c in self.classes:
            print(f'ınıf için en iyi özellikler <{c.upper()}>')
            for w, ratio in sorted(results[c].items(), key = lambda x: x[1], reverse=True)[0:k]:
                print(f'\t{w}\t{ratio}')
            print('')
            
            
if __name__ == '__main__':
    target = 'relevant'

    clf = NaiveBayesClassifier(train_dir = 'haiti/train')
    clf.train()
    print(f'Sınıftaki performans <{target.upper()}>, yasak kelimeleri tutmak')
    precision, recall, f1_score = clf.evaluate(test_dir = 'haiti/dev', target = target)
    print(f'\tHassas: {precision}\t Hatırlamak: {recall}\t F1: {f1_score}\n')
    
    clf = NaiveBayesClassifier(train_dir = 'haiti/train', REMOVE_STOPWORDS=True)
    clf.train()
    print(f'Sınıftaki performans <{target.upper()}>, engellenecek kelimeleri kaldırmak')
    precision, recall, f1_score = clf.evaluate(test_dir = 'haiti/dev', target = target)
    print(f'\tPrecision: {precision}\t Recall: {recall}\t F1: {f1_score}\n')


    clf.print_top_features()


