from __future__ import print_function
import json
#import annotate.py
import re

import math

from argparse import ArgumentParser
from tqdm import tqdm
from lib.dbengine import DBEngine
from lib.query import Query
from lib.common import count_lines

import tensorflow as tf
from seq2seq.encoders import rnn_encoder

import numpy as np
import ast 
import helpers
import matplotlib.pyplot as plt
import tokenize,token
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, MultiRNNCell, DeviceWrapper, ResidualWrapper
from tensorflow.python.layers import core as layers_core
import MySQLdb

w2v_loc = "/data/corpora/glove/glove.6B.100d.txt"
schema_re = re.compile(r'\((.+)\)')
num_re = re.compile(r'[-+]?\d*\.\d+|\d+')
w2v={}

def load_word2vec(file_loc):
    with open(file_loc) as w2v_file:
        grades = []
        for ls in tqdm(w2v_file, total=count_lines(file_loc)):
            lv = ls.split(' ')
            word = lv[0]
            vec = []
            for dim in range(len(lv)-1):
                vec.append(lv[dim+1])
            w2v[word] = vec
def load_seq2sql_file(train_file):
    i=0
    data =[]
    with open(train_file) as lf:
        for ls in tqdm(lf, total=count_lines(train_file)):
            data.append(ast.literal_eval(ls.replace(' ','')))
            i = i + 1
    return data
def index_data(data_file):
    id_ = {0:'',1:'<EOS>', 2:'max', 3:'min', 4:'count', 5:'sum', 6:'avg',7:'=', 8:'>', 9:'<', 10:'op',11:'select', 12:'where', 13:'and', 14:'col', 15:'table', 16:'caption', 17:'page', 18:'section', 19:'cond', 20:'question', 21:'agg', 22:'aggops', 23:'condops',24:'col0',25:'col1',26:'col2',27:'col3',28:'col4',29:'col5',30:'col6',31:'col7',32:'col8',33:'col9',34:'col10',35:'col11',36:'col12',37:'col13',38:'col14',39:'col15',40:'svaha'}
    inv_ = {v:k for k,v in id_.items()}
    i=41
    for df in data_file:
        if 'tables' in df:
            with open(df) as ind:
                for ls in tqdm(ind, total=count_lines(df)):
                    eg = json.loads(ls)
                    if(eg.get('id')):
                        table_id = 'table_{}'.format(eg['id'].replace('-', '_'))
                        if table_id not in inv_:
                            id_[i] =table_id
                            inv_[table_id] =i
                            i = i + 1

        else:
            with open(df) as ind:
                for ls in tqdm(ind, total=count_lines(df)):
                    for t in re.split(r'(,|;|/|/\|!|@|#|$|\"|\(|\)|`|=|\s)\s*',ls):
                        w = str.lower(t)
                        if w not in inv_:
                            id_[i] =w
                            inv_[w] =i
                            i = i + 1
    return id_, inv_
    
def data_(inv_,args):
    agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
    cond_ops = ['=', '>', '<', 'OP']
    syms = ['SELECT', 'WHERE', 'AND', 'COL', 'TABLE', 'CAPTION', 'PAGE', 'SECTION', 'OP', 'COND', 'QUESTION', 'AGG', 'AGGOPS', 'CONDOPS']
    db = MySQLdb.connect(host="localhost",  # your host
                     user="clair",       # username
                     passwd="cfhoCPkr",     # password
                     db="wikisql")   # name of the database
    data = []
    cur = db.cursor()
    table_file = args.source_file.split('.')[0] + '.tables.'+ args.source_file.split('.')[1]
    table_header={}
    
    with open(table_file) as tb:
        grades = []
        for ls in tqdm(tb, total=count_lines(table_file)):
            eg = json.loads(ls)
            table_header[eg['id']] = [re.split(b'[\/|,|;|\s]\s*',h.encode('utf8')) for h in eg['header']]
    with open(args.source_file) as fs:
        grades = []
        for ls in tqdm(fs, total=count_lines(args.source_file)):
            eg = json.loads(ls)
            table_id = eg['table_id']
            cq =''
            i=0
            for cond in eg['sql']['conds']:
                col = cond[0]
                op = cond[1] 
                if isinstance(cond[2],str):
                    cval = cond[2].encode('utf8')
                else:
                    cval = str(cond[2]).encode('utf8')
                cq= cq+('col'+str(col))+cond_ops[op]+'\''+str(cval)+'\' '
                if i< len(eg['sql']['conds'])-1:
                    i = i+1
                    cq = cq + 'AND '

            if(eg['sql']['agg']>0):
                agg = agg_ops[eg['sql']['agg']]
                query = 'SELECT '+ agg+'(col'+str(eg['sql']['sel'])+')'+ ' FROM table_{}'.format((table_id.replace('-','_'))) + ' WHERE '+cq
            else:
                query = 'SELECT '+ ('col'+str(eg['sql']['sel']))+ ' FROM table_{}'.format((table_id.replace('-','_'))) + ' WHERE '+cq
            query = query.replace(str(u'\u2013'.encode('utf8')), '-').replace('\xc4\x81','a').replace('\xe1\xb9\x83','m').replace('[[','').replace(',','').replace('||','').replace(']]','').replace('|','')
            #gold = cur.execute(query)
            ## TO DO .. compare pred with gold
            hl =[] # header list
            v = [r for r in table_header[table_id.replace('table_','').replace('_','-')]]
            for a in v:
                for ap in a:
                    t = re.sub(r'(\(|\)|,|;|!)','',str.lower(str(ap)))
                    if t in inv_:
                        hl.append(inv_[t])
            
            qv = re.split(r'[\(|\)|!|,|?|\-.|\\|\/|\{|\}|\[|\]|#|$|\&|\s]\s*',str.lower(str(eg['question'].encode('UTF8'))))
            qvu = []
            for w in qv:
                if w in inv_ and w not in qvu:
                    qvu.append(w)
            qs= [q for q in re.split(r'(\(|\)|\'|\=|\/|\s)\s*',query)]
            for q in qs:
                q = str.lower(q)
                if q not in inv_:
                    inv_[q]=len(inv_)
                    id_[len(inv_)] = q
            input_ = hl+[inv_[w] for w in qvu] + [inv_[str.lower(s)] for s in syms]
            data.append((input_,[inv_[str.lower(q)] for q in re.split(r'[\(|\)|\'|=|/|\s]\s*',query)]))
    return data

def train(id_,inv_,x_):
    agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
    cond_ops = ['=', '>', '<', 'OP']
    syms = ['SELECT', 'WHERE', 'AND', 'COL', 'TABLE', 'CAPTION', 'PAGE', 'SECTION', 'OP', 'COND', 'QUESTION', 'AGG', 'AGGOPS', 'CONDOPS']

    PAD = 0
    EOS = 1

    vocab_size = len(id_) # 17
    input_embedding_size = 20

    encoder_hidden_units = 20
    decoder_hidden_units = encoder_hidden_units#*2

    encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
    encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')

    decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')

    embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)

    encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)

    encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)

    ################################## attention model #############################
    loss_tracks = dict()

    def do_train(session, model):
        return train_on_copy_task(session, model,length_from=3, length_to=8,vocab_lower=2, vocab_upper=10,batch_size=100,max_batches=5000,batches_in_epoch=1000,verbose=False)
    
    def make_model(**kwa):
        args = dict(cell_class=LSTMCell,
                num_units_encoder=10,
                vocab_size=10,
                embedding_size=10,
                attention=False,
                bidirectional=False,
                debug=False)
        
        args.update(kwa)
        cell_class = args.pop('cell_class')
        num_units_encoder = args.pop('num_units_encoder')
        num_units_decoder = num_units_encoder

        if args['bidirectional']:
            num_units_decoder *= 2

        args['encoder_cell'] = cell_class(num_units_encoder)
        args['decoder_cell'] = cell_class(num_units_decoder)
        return Seq2SeqModel(**args)
    
    tf.reset_default_graph()
    tf.set_random_seed(1)
    with tf.Session() as session:
        model = make_model(bidirectional=False, attention=True)
        session.run(tf.global_variables_initializer())
        loss_tracks['forward encoder, with attention'] = do_train(session, model)
    
    #################################### train #########################################
    batch_size = 4
    batches = helpers.random_sequences(length_from=3, length_to=8,vocab_lower=2, vocab_upper=10,batch_size=batch_size)
    sql_batches = [i[1] for i in x_[1:20]]

    print('head of the batch:')
    for seq in sql_batches[1:5]:
        print([id_[s] for s in seq])
    def next_feed():
        batch = sql_batches
        encoder_inputs_, encoder_input_lengths_ = helpers.batch(batch)
        decoder_targets_, _ = helpers.batch([(sequence) + [EOS] + [PAD] * 2 for sequence in batch])
        return {encoder_inputs: encoder_inputs_,encoder_inputs_length: encoder_input_lengths_,decoder_targets: decoder_targets_,}
    
    loss_track = []
    max_batches = 200
    batches_in_epoch = 20
    try:
        for batch in range(max_batches):
            fd = next_feed()
            _, l = sess.run([train_op, loss], fd)
            loss_track.append(l)

            if batch == 0 or batch % batches_in_epoch == 0:
                print('batch {}'.format(batch))
                print('  minibatch loss: {}'.format(sess.run(loss, fd)))
                predict_ = sess.run(decoder_prediction, fd)
                for i, (inp, pred) in enumerate(zip(fd[encoder_inputs].T, predict_.T)):
                    print('  sample {}:'.format(i + 1))
                    print('    input     > {}'.format([id_[q] for q in inp]))
                    print('    predicted > {}'.format([id_[q] for q in pred]))
                    if i >= 2:
                        break
                print()
    except KeyboardInterrupt:
        print('training interrupted')
    plt.plot(loss_track)
    plt.savefig('loss track')
    print('loss {:.4f} after {} examples (batch_size={})'.format(loss_track[-1], len(loss_track)*batch_size, batch_size))



if __name__ == '__main__':
    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    parser = ArgumentParser()
    parser.add_argument('source_file', help='source file for the prediction')
    parser.add_argument('db_file', help='source database for the prediction')
    args = parser.parse_args()
    table_file = args.source_file.split('.')[0] + '.tables.'+ args.source_file.split('.')[1]

    id_,inv_ = index_data([args.source_file,table_file]) # index input data 
    x_ =data_(inv_,args) # build feature vector for training
    train(id_,inv_,x_) # start training
