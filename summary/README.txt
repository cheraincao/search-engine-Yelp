1.Dependencies
python==3.6
numpy==1.16.1
gensim==3.8.1
Keras==2.4.3
Theano==1.0.5
nltk==3.4.3
scipy
mkl-service
libpython
m2w64-toolchain
pathlib

2.get skipthoughts ready

You will first need to download the model files and word embeddings. The embedding files (utable and btable) are quite large (>2GB) so make sure there is enough space available. The encoder vocabulary can be found in dictionary.txt.

    wget http://www.cs.toronto.edu/~rkiros/models/dictionary.txt
    wget http://www.cs.toronto.edu/~rkiros/models/utable.npy
    wget http://www.cs.toronto.edu/~rkiros/models/btable.npy
    wget http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz
    wget http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz.pkl
    wget http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz
    wget http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz.pkl

Once these are downloaded, open skipthoughts.py and set the paths to the above files (path_to_models and path_to_tables)
for example, I saved the above downloading files in model folder, then set path as below:
path_to_models = 'D:/python/summary/models/'
path_to_tables = 'D:/python/summary/models/'

4.Dataset business_tip is a combine version of yelp_academic_dataset_business and yelp_academic_dataset_tip from yelp dataset.
We extract 'text' feature from yelp_academic_dataset_tip dataset and use business_id to match it with yelp_academic_dataset_business dataset.
then we create a new feature 'tip_text', which will be used to generate summary.

5.Run summary.py to generate static summary. Other files are using to import skipthoughts, which can be ignored.

Note: Because skipthoughts.py was set up under python==2.7 environment, some steps may need to avoid error before running summary.py
1.Add parentheses for all the 'print' in skipthoughts.py
2.change 'import cPickle as pkl' in skipthoughts.py to 'import pickle as pkl'
3.File skipthoughts.py, line 252, change 'iteritems' to 'items'
4.File skipthoughts.py, line 129, numbatches = len(ds[k]) // batch_size + 1
5.File skipthoughts.py, line 79,80 add numpy.load options allow_pickle=True, encoding="latin1"